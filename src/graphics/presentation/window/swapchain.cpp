#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "gpu_blit_shaders/gpu_blit_fs_triangle_spv.h"
#include "gpu_blit_shaders/gpu_upscale_bicubic_spv.h"
#include "gpu_blit_shaders/gpu_upscale_fsr_spv.h"
#include "gpu_blit_shaders/gpu_upscale_scaleforce_spv.h"
#include "gpu_blit_shaders/gpu_sharpen_rcas_spv.h"
#include "gpu_blit_shaders/gpu_video_out_overlay_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window/windowInternal.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
#include <vulkan/vk_platform.h>

// IWYU pragma: no_include <intrin.h>

namespace Libs::Graphics {

struct Presenter::Frame {
        VulkanImage   image;
        vk::ImageView view         = nullptr;
        uint64_t      present_tick = 0;
        bool          busy         = false;
        bool          hdr          = false;
        bool          blank        = false;

        void Configure(GraphicContext& graphics, vk::Extent2D extent, vk::Format format);
        void Transit(vk::CommandBuffer command, vk::ImageLayout layout, vk::AccessFlags2 access);
        void CopyFrom(CommandBuffer& command, Image& source);
        void Clear(CommandBuffer& command, const vk::ClearColorValue& color);
};

class FramePool final {
public:
        FramePool(WindowContext& window, CommandScheduler& scheduler)
            : m_window(window), m_scheduler(scheduler) {}
        ~FramePool() {
                m_scheduler.Wait(m_scheduler.CurrentTick() - 1);
                for (auto& frame: m_frames) {
                        m_window.graphic_ctx.device.destroyImageView(frame->view, nullptr);
                        if (frame->image.image != nullptr) {
                                m_window.graphic_ctx.DeleteImage(frame->image);
                        }
                }
        }
        KYTY_CLASS_NO_COPY(FramePool);

        void Initialize(uint32_t count, vk::Format format) {
                if (count == 0 || format == vk::Format::eUndefined) {
                        EXIT("prepared-frame pool requires at least one frame\n");
                }
                Common::LockGuard lock(m_mutex);
                if (!m_frames.empty()) {
                        EXIT("prepared-frame pool was initialized twice\n");
                }
                m_format = format;
                m_frames.reserve(count);
                m_free.reserve(count);
                for (uint32_t i = 0; i < count; i++) {
                        auto frame = std::make_unique<Presenter::Frame>();
                        m_free.push_back(frame.get());
                        m_frames.push_back(std::move(frame));
                }
        }

        void SetFormat(vk::Format format) {
                if (format == vk::Format::eUndefined) {
                        EXIT("prepared-frame pool requires a presentation format\n");
                }
                Common::LockGuard lock(m_mutex);
                m_format = format;
        }

        vk::Format GetFormat() {
                Common::LockGuard lock(m_mutex);
                if (m_format == vk::Format::eUndefined) {
                        EXIT("prepared-frame pool has no presentation format\n");
                }
                return m_format;
        }

        Presenter::Frame* Acquire(vk::Extent2D extent, vk::Format format) {
                m_mutex.Lock();
                if (m_frames.empty()) {
                        EXIT("prepared-frame pool was used before swapchain initialization\n");
                }
                // A synchronized flip may need more frames than the swapchain has images.
                // Queue capacity bounds growth; waiting here can prevent its master from recording.
                if (m_free.empty()) {
                        auto frame = std::make_unique<Presenter::Frame>();
                        m_free.push_back(frame.get());
                        m_frames.push_back(std::move(frame));
                }
                const auto compatible = std::ranges::find_if(m_free, [&](const auto* frame) {
                        return frame->image.extent.width == extent.width &&
                               frame->image.extent.height == extent.height && frame->image.format == format;
                });
                auto*      frame      = compatible == m_free.end() ? m_free.back() : *compatible;
                if (compatible != m_free.end()) {
                        *compatible = m_free.back();
                }
                m_free.pop_back();
                if (frame->busy) {
                        EXIT("prepared-frame pool returned an invalid frame\n");
                }
                frame->busy = true;
                m_mutex.Unlock();

                m_scheduler.Wait(frame->present_tick);
                return frame;
        }

        void ValidateForPresent(Presenter::Frame* frame) {
                Common::LockGuard lock(m_mutex);
                if (frame == nullptr || !frame->busy) {
                        EXIT("prepared frame has invalid presentation ownership\n");
                }
        }

        void Release(Presenter::Frame* frame) {
                if (frame == nullptr) {
                        EXIT("cannot release a null prepared frame\n");
                }
                Common::LockGuard lock(m_mutex);
                if (!frame->busy) {
                        EXIT("prepared frame was released twice\n");
                }
                frame->busy = false;
                m_free.push_back(frame);
        }

private:
        WindowContext&                                 m_window;
        CommandScheduler&                              m_scheduler;
        Common::Mutex                                  m_mutex;
        std::vector<std::unique_ptr<Presenter::Frame>> m_frames;
        std::vector<Presenter::Frame*>                 m_free;
        vk::Format                                     m_format = vk::Format::eUndefined;
};

void Presenter::Frame::Configure(GraphicContext& graphics, vk::Extent2D extent, vk::Format format) {
        if (extent.width == 0 || extent.height == 0 || format == vk::Format::eUndefined) {
                EXIT("unsupported prepared frame, extent=%ux%u format=%d\n", extent.width, extent.height,
                     static_cast<int>(format));
        }
        auto&      dst        = image;
        const bool compatible = dst.image != nullptr && dst.extent.width == extent.width &&
                                dst.extent.height == extent.height && dst.format == format;
        if (compatible) {
                return;
        }

        const auto features = graphics.GetFormatProperties(format).optimalTilingFeatures;
        const auto required =
            vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eSampledImageFilterLinear |
            vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eTransferSrc |
            vk::FormatFeatureFlagBits::eTransferDst;
        if ((features & required) != required) {
                EXIT("prepared presentation format lacks optimal blit support: format=%d features=0x%x\n",
                     static_cast<int>(format), static_cast<vk::FormatFeatureFlags::MaskType>(features));
        }

        if (dst.image != nullptr) {
                graphics.device.destroyImageView(view, nullptr);
                view = nullptr;
                graphics.DeleteImage(dst);
        }

        vk::ImageCreateInfo create {};
        create.sType         = vk::StructureType::eImageCreateInfo;
        create.imageType     = vk::ImageType::e2D;
        create.extent        = {extent.width, extent.height, 1};
        create.mipLevels     = 1;
        create.arrayLayers   = 1;
        create.format        = format;
        create.tiling        = vk::ImageTiling::eOptimal;
        create.initialLayout = vk::ImageLayout::eUndefined;
        create.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
                       vk::ImageUsageFlagBits::eSampled;
        create.sharingMode = vk::SharingMode::eExclusive;
        create.samples     = vk::SampleCountFlagBits::e1;
        if (!graphics.CreateImage(create, dst)) {
                EXIT("failed to allocate prepared presentation image, extent=%ux%u format=%d\n",
                     extent.width, extent.height, static_cast<int>(format));
        }
}

void Presenter::Frame::Transit(vk::CommandBuffer command, vk::ImageLayout layout,
                               vk::AccessFlags2 access) {
        const auto     stage  = access == vk::AccessFlagBits2::eTransferRead ||
                                        access == vk::AccessFlagBits2::eTransferWrite
                                    ? vk::PipelineStageFlagBits2::eTransfer
                                    : vk::PipelineStageFlagBits2::eAllCommands;
        constexpr auto writes = vk::AccessFlagBits2::eTransferWrite |
                                vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eMemoryWrite;
        if (image.state.layout == layout && image.state.access_mask == access &&
            !static_cast<bool>(image.state.access_mask & writes)) {
                return;
        }
        vk::ImageMemoryBarrier2 barrier {};
        barrier.srcStageMask                    = image.state.pl_stage;
        barrier.srcAccessMask                   = image.state.access_mask;
        barrier.dstStageMask                    = stage;
        barrier.dstAccessMask                   = access;
        barrier.oldLayout                       = image.state.layout;
        barrier.newLayout                       = layout;
        barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.image                           = image.image;
        barrier.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
        vk::DependencyInfo dependency {};
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers    = &barrier;
        command.pipelineBarrier2(dependency);
        image.state = {stage, access, layout};
        image.subresource_states.clear();
}

void Presenter::Frame::CopyFrom(CommandBuffer& command_buffer, Image& source) {
        command_buffer.EndRendering();
        auto command = command_buffer.Handle();
        source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
                       command);
        Transit(command, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite);
        vk::ImageCopy copy {};
        copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, source.backing.layers};
        copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, image.layers};
        copy.extent         = {std::min(source.backing.extent.width, image.extent.width),
                               std::min(source.backing.extent.height, image.extent.height), 1};
        EXIT_IF(copy.srcSubresource.layerCount != copy.dstSubresource.layerCount);
        command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, image.image,
                          vk::ImageLayout::eTransferDstOptimal, copy);
}

void Presenter::Frame::Clear(CommandBuffer& command_buffer, const vk::ClearColorValue& color) {
        command_buffer.EndRendering();
        auto command = command_buffer.Handle();
        Transit(command, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite);
        const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        command.clearColorImage(image.image, vk::ImageLayout::eTransferDstOptimal, &color, 1, &range);
}

class Swapchain final {
public:
        enum class Status : uint8_t { Success, Recreate, SurfaceLost };

        explicit Swapchain(WindowContext& window): m_window(window) {}
        ~Swapchain();
        KYTY_CLASS_NO_COPY(Swapchain);

        void                 Create();
        void                 Recreate(bool surface_lost = false);
        [[nodiscard]] bool   NeedsResize() const;
        [[nodiscard]] Status AcquireNextImage();
        [[nodiscard]] bool   PrepareSystemOverlay();
        void     RecordPresentCommands(CommandBuffer& command, Presenter::Frame* source,
                                       const Presenter::Layer& overlay, bool draw_system_overlay);
        uint64_t Submit(CommandScheduler& scheduler);
        [[nodiscard]] Status Present();

        [[nodiscard]] uint32_t ImageCount() const noexcept {
                return static_cast<uint32_t>(m_images.size());
        }
        [[nodiscard]] vk::Format Format() const noexcept { return m_format; }
        [[nodiscard]] bool IsHdrSupported() const noexcept { return m_hdr_supported.load(); }
        bool SetHdr(bool hdr) {
                const bool changed = m_hdr_active != (hdr && IsHdrSupported());
                m_hdr_requested = hdr;
                return changed;
        }

private:
        vk::SurfaceFormatKHR SelectFormat();
        void Destroy();
        void DrawOverlay(vk::CommandBuffer command, const Presenter::Layer& layer);
        void DrawUpscale(vk::CommandBuffer command, Presenter::Frame& source, const vk::Rect2D& rect);
        [[nodiscard]] bool EnsureSharpenTarget();
        [[nodiscard]] vk::Rect2D ComputePresentRect(const vk::Extent2D& src) const;

        WindowContext&   m_window;
        vk::SwapchainKHR m_handle = nullptr;
        vk::Format       m_format = vk::Format::eUndefined;
        std::atomic_bool m_hdr_supported {false};
        bool             m_hdr_requested = false;
        bool             m_hdr_active    = false;
        vk::Extent2D     m_extent {};
        // Drawable pixel size observed when this swapchain was created.
        vk::Extent2D                   m_window_extent {};
        std::vector<vk::Image>         m_images;
        std::vector<vk::ImageView>     m_image_views;
        std::vector<vk::Semaphore>     m_image_acquired;
        std::vector<vk::Semaphore>     m_render_complete;
        std::unique_ptr<SystemOverlay> m_system_overlay;
        vk::DescriptorSetLayout        m_overlay_descriptors = nullptr;
        vk::PipelineLayout             m_overlay_layout      = nullptr;
        vk::Pipeline                   m_overlay_pipeline    = nullptr;
        vk::Sampler                    m_overlay_sampler     = nullptr;
        vk::DescriptorSetLayout        m_upscale_descriptors = nullptr;
        vk::PipelineLayout             m_upscale_layout      = nullptr;
        vk::Pipeline                   m_upscale_pipeline    = nullptr;
        vk::Sampler                    m_upscale_sampler     = nullptr;
        // Second-pass FSR1 RCAS sharpening pipeline. Created on demand when
        // the user enables upscaling sharpening; null otherwise.
        vk::Pipeline                   m_rcas_pipeline       = nullptr;
        // Intermediate color target the upscaling filter renders into before
        // the RCAS pass samples it. Owned here (not per-frame) because it is
        // only needed when sharpening is enabled and tracks the swapchain
        // extent/format.
        Image                          m_sharpen_image {};
        vk::ImageView                  m_sharpen_view        = nullptr;
        // Last known state of the intermediate target for explicit barriers.
        vk::PipelineStageFlags2        m_sharpen_stage       = vk::PipelineStageFlagBits2::eTopOfPipe;
        vk::AccessFlags2               m_sharpen_access      = vk::AccessFlagBits2::eNone;
        vk::ImageLayout                m_sharpen_layout      = vk::ImageLayout::eUndefined;
        // True when the current frame renders the guest image with a graphics
        // pipeline (shader upscaling) instead of a blit; the acquire semaphore
        // wait stage differs between the two paths.
        bool                           m_present_uses_graphics = false;
        uint32_t                       m_image_index         = static_cast<uint32_t>(-1);
        uint32_t                       m_frame_index         = 0;
};

struct Presenter::Impl {
        explicit Impl(WindowContext& owner)
            : renderer(*owner.render_context), window(owner), swapchain(owner),
              present_scheduler(renderer, owner.graphic_ctx), frames(owner, present_scheduler) {
                EXIT_IF(owner.render_context == nullptr);
                swapchain.Create();
                frames.Initialize(swapchain.ImageCount(), swapchain.Format());
        }

        void RecoverSwapchain(Swapchain::Status status) {
                LOGF("Recovering Vulkan swapchain%s\n",
                     status == Swapchain::Status::SurfaceLost ? " and surface" : "");
                swapchain.Recreate(status == Swapchain::Status::SurfaceLost);
                frames.SetFormat(swapchain.Format());
        }

        Image& ResolveSurface(const ImageInfo& info) {
                TextureCache::ImageDesc desc {};
                desc.info                  = info;
                desc.view_info.format      = info.pixel_format;
                desc.view_info.type        = vk::ImageViewType::e2D;
                desc.view_info.aspect      = vk::ImageAspectFlagBits::eColor;
                desc.view_info.base_level  = 0;
                desc.view_info.level_count = 1;
                desc.view_info.base_layer  = 0;
                desc.view_info.layer_count = 1;
                desc.view_info.usage       = vk::ImageUsageFlagBits::eTransferSrc;
                desc.type                  = TextureCache::BindingType::VideoOut;

                auto&      cache      = renderer.GetTextureCache();
                const auto image_id   = cache.FindImage(desc);
                auto&      image      = cache.GetImage(image_id);
                image.usage.video_out = true;
                cache.UpdateImage(image_id);
                return image;
        }
        void Present();

        RenderContext&        renderer;
        WindowContext&        window;
        Swapchain             swapchain;
        CommandScheduler      present_scheduler;
        FramePool             frames;
        Common::Mutex         present_mutex;
        std::array<Layer, 2>  layers {};
        std::atomic<uint64_t> presented_overlay_revision {0};
};

vk::SurfaceFormatKHR Swapchain::SelectFormat() {
        const auto& formats    = m_window.surface_capabilities.formats;
        auto&       graphics   = m_window.graphic_ctx;
        const auto  hdr_format = std::ranges::find_if(formats, [&](const auto& candidate) {
                return candidate.colorSpace == vk::ColorSpaceKHR::eHdr10St2084EXT &&
                       IsHdrSwapchainFormat(candidate.format) &&
                       static_cast<bool>(
                           graphics.GetFormatProperties(candidate.format).optimalTilingFeatures &
                           vk::FormatFeatureFlagBits::eBlitDst);
        });
        m_hdr_supported        = Config::HdrEnabled() && hdr_format != formats.end();
        m_hdr_active           = m_hdr_requested && IsHdrSupported();
        if (m_hdr_active) {
                return *hdr_format;
        }
        if (Config::HdrEnabled() && !IsHdrSupported()) {
                LOGF("HDR10 surface unavailable; using SDR output\n");
        }
        if (formats.size() == 1 && formats.front().format == vk::Format::eUndefined) {
                return {vk::Format::eR8G8B8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
        }
        for (const auto& candidate: formats) {
                if (candidate.colorSpace != vk::ColorSpaceKHR::eSrgbNonlinear) {
                        continue;
                }
                switch (candidate.format) {
                        case vk::Format::eB8G8R8A8Unorm:
                        case vk::Format::eR8G8B8A8Unorm: return candidate;
                        default: break;
                }
        }
        EXIT("no supported UNORM swapchain format\n");
}

void Swapchain::Create() {
        auto& graphics = m_window.graphic_ctx;
        EXIT_IF(graphics.device == nullptr);
        EXIT_IF(m_window.surface == nullptr);

        // Capture the size before querying surface capabilities. If the window changes
        // during creation, the next presentation will detect the newer size.
        {
                Common::LockGuard lock(m_window.mutex);
                m_window_extent = {graphics.screen_width, graphics.screen_height};
        }
        EXIT_IF(m_window_extent.width == 0);
        EXIT_IF(m_window_extent.height == 0);
        m_window.RefreshSurfaceCapabilities();
        const auto& surface = m_window.surface_capabilities;
        EXIT_NOT_IMPLEMENTED(surface.formats.empty());

        m_extent = surface.capabilities.currentExtent;
        if (m_extent.width == std::numeric_limits<uint32_t>::max()) {
                m_extent.width =
                    std::clamp(m_window_extent.width, surface.capabilities.minImageExtent.width,
                               surface.capabilities.maxImageExtent.width);
                m_extent.height =
                    std::clamp(m_window_extent.height, surface.capabilities.minImageExtent.height,
                               surface.capabilities.maxImageExtent.height);
        }
        uint32_t image_count = surface.capabilities.minImageCount + 1;
        if (const uint32_t requested = Config::GetMaxQueuedFrames(); requested != 0) {
                image_count = requested;
        }
        if (surface.capabilities.maxImageCount != 0) {
                image_count = std::min(image_count, surface.capabilities.maxImageCount);
        }
        image_count = std::max(image_count, surface.capabilities.minImageCount);
        const auto transform =
            surface.capabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity
                ? vk::SurfaceTransformFlagBitsKHR::eIdentity
                : surface.capabilities.currentTransform;
        const auto composite =
            surface.capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque
                ? vk::CompositeAlphaFlagBitsKHR::eOpaque
                : vk::CompositeAlphaFlagBitsKHR::eInherit;

        const auto format = SelectFormat();
        m_format                      = format.format;
        const auto swapchain_features = graphics.GetFormatProperties(m_format).optimalTilingFeatures;
        if (!static_cast<bool>(swapchain_features & vk::FormatFeatureFlagBits::eBlitDst)) {
                EXIT("swapchain format cannot be a blit destination: format=%d\n",
                     static_cast<int>(m_format));
        }

        vk::SwapchainCreateInfoKHR create_info {};
        create_info.sType            = vk::StructureType::eSwapchainCreateInfoKHR;
        create_info.surface          = m_window.surface;
        create_info.minImageCount    = image_count;
        create_info.imageFormat      = format.format;
        create_info.imageColorSpace  = format.colorSpace;
        create_info.imageExtent      = m_extent;
        create_info.imageArrayLayers = 1;
        create_info.imageUsage =
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
        create_info.imageSharingMode = vk::SharingMode::eExclusive;
        create_info.preTransform     = transform;
        create_info.compositeAlpha   = composite;
        switch (Config::GetPresentMode()) {
                case Config::PresentMode::Mailbox:
                        create_info.presentMode = vk::PresentModeKHR::eMailbox;
                        break;
                case Config::PresentMode::Immediate:
                        create_info.presentMode = vk::PresentModeKHR::eImmediate;
                        break;
                case Config::PresentMode::Fifo:
                default: create_info.presentMode = vk::PresentModeKHR::eFifo; break;
        }
        if (std::find(surface.present_modes.begin(), surface.present_modes.end(),
                      create_info.presentMode) == surface.present_modes.end()) {
                LOGF("warning: requested present mode is unavailable; falling back to Fifo\n");
                create_info.presentMode = vk::PresentModeKHR::eFifo;
        }
        create_info.clipped = VK_TRUE;
        RequireVulkanSuccess(graphics.device.createSwapchainKHR(&create_info, nullptr, &m_handle),
                             "vkCreateSwapchainKHR");
        EXIT_IF(m_handle == nullptr);
        LOGF("Vulkan presentation: %s\n", m_hdr_active ? "HDR10 (PQ)" : "SDR");

        m_images = EnumerateVulkan<vk::Image>(
            "vkGetSwapchainImagesKHR", [&](uint32_t* count, vk::Image* images) {
                    return graphics.device.getSwapchainImagesKHR(m_handle, count, images);
            });
        EXIT_NOT_IMPLEMENTED(m_images.empty());

        m_image_views.resize(m_images.size());
        for (size_t i = 0; i < m_images.size(); i++) {
                vk::ImageViewCreateInfo view {};
                view.sType                           = vk::StructureType::eImageViewCreateInfo;
                view.image                           = m_images[i];
                view.viewType                        = vk::ImageViewType::e2D;
                view.format                          = m_format;
                view.components                      = {};
                view.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
                view.subresourceRange.baseArrayLayer = 0;
                view.subresourceRange.baseMipLevel   = 0;
                view.subresourceRange.layerCount     = 1;
                view.subresourceRange.levelCount     = 1;
                RequireVulkanSuccess(graphics.device.createImageView(&view, nullptr, &m_image_views[i]),
                                     "vkCreateImageView");
                EXIT_IF(m_image_views[i] == nullptr);
        }

        vk::SemaphoreCreateInfo semaphore_info {};
        semaphore_info.sType = vk::StructureType::eSemaphoreCreateInfo;
        m_image_acquired.resize(m_images.size());
        m_render_complete.resize(m_images.size());
        for (size_t i = 0; i < m_images.size(); i++) {
                RequireVulkanSuccess(
                    graphics.device.createSemaphore(&semaphore_info, nullptr, &m_image_acquired[i]),
                    "create swapchain image-acquired semaphore");
                RequireVulkanSuccess(
                    graphics.device.createSemaphore(&semaphore_info, nullptr, &m_render_complete[i]),
                    "create swapchain render-complete semaphore");
        }
        m_image_index = static_cast<uint32_t>(-1);
        m_frame_index = 0;
}

Swapchain::~Swapchain() {
        Destroy();
}

void Swapchain::Destroy() {
        if (m_handle == nullptr && m_image_acquired.empty() && m_render_complete.empty() &&
            m_image_views.empty()) {
                return;
        }
        auto& graphics = m_window.graphic_ctx;

        {
                Common::LockGuard queue_lock(graphics.queue_mutex);
                RequireVulkanSuccess(graphics.queue.waitIdle(), "wait for swapchain queue");
        }
        if (m_system_overlay != nullptr) {
                m_system_overlay->ReleaseVulkan();
        }
        graphics.device.destroyPipeline(m_overlay_pipeline, nullptr);
        graphics.device.destroyPipelineLayout(m_overlay_layout, nullptr);
        graphics.device.destroyDescriptorSetLayout(m_overlay_descriptors, nullptr);
        graphics.device.destroySampler(m_overlay_sampler, nullptr);
        m_overlay_pipeline    = nullptr;
        m_overlay_layout      = nullptr;
        m_overlay_descriptors = nullptr;
        m_overlay_sampler     = nullptr;

        graphics.device.destroyPipeline(m_upscale_pipeline, nullptr);
        graphics.device.destroyPipelineLayout(m_upscale_layout, nullptr);
        graphics.device.destroyDescriptorSetLayout(m_upscale_descriptors, nullptr);
        graphics.device.destroySampler(m_upscale_sampler, nullptr);
        m_upscale_pipeline    = nullptr;
        m_upscale_layout      = nullptr;
        m_upscale_descriptors = nullptr;
        m_upscale_sampler     = nullptr;

        graphics.device.destroyPipeline(m_rcas_pipeline, nullptr);
        m_rcas_pipeline = nullptr;
        if (m_sharpen_view != nullptr) {
                graphics.device.destroyImageView(m_sharpen_view, nullptr);
                m_sharpen_view = nullptr;
        }
        if (m_sharpen_image.image != nullptr) {
                graphics.DeleteImage(m_sharpen_image);
                m_sharpen_image.image = nullptr;
        }
        m_sharpen_stage  = vk::PipelineStageFlagBits2::eTopOfPipe;
        m_sharpen_access = vk::AccessFlagBits2::eNone;
        m_sharpen_layout = vk::ImageLayout::eUndefined;

        for (const auto semaphore: m_image_acquired) {
                if (semaphore != nullptr) {
                        graphics.device.destroySemaphore(semaphore, nullptr);
                }
        }
        for (const auto semaphore: m_render_complete) {
                if (semaphore != nullptr) {
                        graphics.device.destroySemaphore(semaphore, nullptr);
                }
        }
        for (const auto view: m_image_views) {
                if (view != nullptr) {
                        graphics.device.destroyImageView(view, nullptr);
                }
        }
        if (m_handle != nullptr) {
                graphics.device.destroySwapchainKHR(m_handle, nullptr);
        }

        m_handle        = nullptr;
        m_format        = vk::Format::eUndefined;
        m_extent        = {};
        m_window_extent = {};
        m_image_index   = static_cast<uint32_t>(-1);
        m_frame_index   = 0;
        m_images.clear();
        m_image_views.clear();
        m_image_acquired.clear();
        m_render_complete.clear();
}

void Swapchain::Recreate(bool surface_lost) {
        Destroy();
        if (surface_lost) {
#if defined(__APPLE__)
                // Surface recreation goes through SDL_Vulkan_CreateSurface, which touches the
                // window's view/layer and must run on the main thread on macOS.
                EXIT_IF(!SDL_RunOnMainThread(
                    [](void* window) { static_cast<WindowContext*>(window)->RecreateSurface(); }, &m_window,
                    true));
#else
                m_window.RecreateSurface();
#endif
        }
        Create();
}

bool Swapchain::NeedsResize() const {
        Common::LockGuard lock(m_window.mutex);
        return m_window_extent.width != m_window.graphic_ctx.screen_width ||
               m_window_extent.height != m_window.graphic_ctx.screen_height;
}

Swapchain::Status Swapchain::AcquireNextImage() {
        EXIT_IF(m_handle == nullptr || m_frame_index >= m_image_acquired.size());
        m_image_index     = static_cast<uint32_t>(-1);
        const auto result = m_window.graphic_ctx.device.acquireNextImageKHR(
            m_handle, std::numeric_limits<uint64_t>::max(), m_image_acquired[m_frame_index], nullptr,
            &m_image_index);
        switch (result) {
                case vk::Result::eSuccess: break;
                case vk::Result::eSuboptimalKHR:
                        LOGF("vkAcquireNextImageKHR returned vk::Result::eSuboptimalKHR\n");
                        return Status::Recreate;
                case vk::Result::eErrorOutOfDateKHR:
                        LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorOutOfDateKHR\n");
                        return Status::Recreate;
                case vk::Result::eErrorUnknown:
                        LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorUnknown\n");
                        return Status::Recreate;
                case vk::Result::eErrorSurfaceLostKHR:
                        LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorSurfaceLostKHR\n");
                        return Status::SurfaceLost;
                default: EXIT("vkAcquireNextImageKHR failed: %s\n", vk::to_string(result).c_str());
        }
        EXIT_IF(m_image_index >= m_images.size());
        return Status::Success;
}

bool Swapchain::PrepareSystemOverlay() {
        if (m_system_overlay == nullptr) {
                m_system_overlay = std::make_unique<SystemOverlay>(m_window.graphic_ctx);
        }
        return m_system_overlay->PrepareFrame(m_extent, m_format, ImageCount());
}

void Swapchain::DrawOverlay(vk::CommandBuffer command, const Presenter::Layer& layer) {
        auto device = m_window.graphic_ctx.device;
        if (m_overlay_pipeline == nullptr) {
                vk::SamplerCreateInfo sampler {};
                sampler.magFilter    = vk::Filter::eLinear;
                sampler.minFilter    = vk::Filter::eLinear;
                sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
                sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
                sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
                RequireVulkanSuccess(device.createSampler(&sampler, nullptr, &m_overlay_sampler),
                                     "create video-out overlay sampler");
                const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eCombinedImageSampler,
                                                              1, vk::ShaderStageFlagBits::eFragment};
                vk::DescriptorSetLayoutCreateInfo    descriptors {};
                descriptors.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
                descriptors.bindingCount = 1;
                descriptors.pBindings    = &binding;
                RequireVulkanSuccess(
                    device.createDescriptorSetLayout(&descriptors, nullptr, &m_overlay_descriptors),
                    "create video-out overlay descriptor layout");
                const vk::PushConstantRange alpha {vk::ShaderStageFlagBits::eFragment, 0, sizeof(uint32_t)};
                vk::PipelineLayoutCreateInfo layout {};
                layout.setLayoutCount         = 1;
                layout.pSetLayouts            = &m_overlay_descriptors;
                layout.pushConstantRangeCount = 1;
                layout.pPushConstantRanges    = &alpha;
                RequireVulkanSuccess(device.createPipelineLayout(&layout, nullptr, &m_overlay_layout),
                                     "create video-out overlay pipeline layout");
                const auto vertex   = CompileSPV(GPU_BLIT_FS_TRIANGLE_SPV, device);
                const auto fragment = CompileSPV(GPU_VIDEO_OUT_OVERLAY_SPV, device);
                std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
                stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
                stages[0].module = vertex;
                stages[0].pName  = "main";
                stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
                stages[1].module = fragment;
                stages[1].pName  = "main";
                vk::PipelineVertexInputStateCreateInfo   vertex_input {};
                vk::PipelineInputAssemblyStateCreateInfo assembly {};
                assembly.topology = vk::PrimitiveTopology::eTriangleList;
                vk::PipelineViewportStateCreateInfo viewport {};
                viewport.viewportCount = 1;
                viewport.scissorCount  = 1;
                vk::PipelineRasterizationStateCreateInfo rasterization {};
                rasterization.lineWidth = 1.0f;
                vk::PipelineMultisampleStateCreateInfo multisample {};
                multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;
                vk::PipelineColorBlendAttachmentState attachment {};
                attachment.blendEnable         = VK_TRUE;
                attachment.srcColorBlendFactor = vk::BlendFactor::eOne;
                attachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
                attachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
                attachment.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
                attachment.colorWriteMask = vk::ColorComponentFlagBits::eR |
                                            vk::ColorComponentFlagBits::eG |
                                            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
                vk::PipelineColorBlendStateCreateInfo blend {};
                blend.attachmentCount = 1;
                blend.pAttachments    = &attachment;
                const std::array dynamic_states {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
                vk::PipelineDynamicStateCreateInfo dynamic {};
                dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
                dynamic.pDynamicStates    = dynamic_states.data();
                vk::PipelineRenderingCreateInfo rendering {};
                rendering.colorAttachmentCount    = 1;
                rendering.pColorAttachmentFormats = &m_format;
                vk::GraphicsPipelineCreateInfo create {};
                create.pNext               = &rendering;
                create.stageCount          = static_cast<uint32_t>(stages.size());
                create.pStages             = stages.data();
                create.pVertexInputState   = &vertex_input;
                create.pInputAssemblyState = &assembly;
                create.pViewportState      = &viewport;
                create.pRasterizationState = &rasterization;
                create.pMultisampleState   = &multisample;
                create.pColorBlendState    = &blend;
                create.pDynamicState       = &dynamic;
                create.layout              = m_overlay_layout;
                RequireVulkanSuccess(
                    device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &m_overlay_pipeline),
                    "create video-out overlay pipeline");
                device.destroyShaderModule(fragment, nullptr);
                device.destroyShaderModule(vertex, nullptr);
        }
        auto& frame = *layer.frame;
        if (frame.view == nullptr) {
                vk::ImageViewCreateInfo view {};
                view.image            = frame.image.image;
                view.viewType         = vk::ImageViewType::e2D;
                view.format           = frame.image.format;
                view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
                RequireVulkanSuccess(device.createImageView(&view, nullptr, &frame.view),
                                     "create video-out overlay image view");
        }
        const vk::DescriptorImageInfo image {m_overlay_sampler, frame.view,
                                             vk::ImageLayout::eShaderReadOnlyOptimal};
        vk::WriteDescriptorSet        write {};
        write.dstBinding      = 0;
        write.descriptorCount = 1;
        write.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
        write.pImageInfo      = &image;
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, m_overlay_layout, 0, 1, &write);
        const uint32_t flags = (layer.premultiplied_alpha ? 1u : 0u) |
                               (m_hdr_active ? 2u : 0u) | (frame.hdr ? 4u : 0u) |
                               (frame.image.format == vk::Format::eR16G16B16A16Sfloat ? 8u : 0u);
        command.pushConstants(m_overlay_layout, vk::ShaderStageFlagBits::eFragment, 0,
                              sizeof(flags), &flags);
        command.bindPipeline(vk::PipelineBindPoint::eGraphics, m_overlay_pipeline);
        vk::RenderingAttachmentInfo attachment {};
        attachment.imageView   = m_image_views[m_image_index];
        attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
        attachment.storeOp     = vk::AttachmentStoreOp::eStore;
        vk::RenderingInfo rendering {};
        rendering.renderArea.extent    = m_extent;
        rendering.layerCount           = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments    = &attachment;
        command.beginRendering(&rendering);
        const vk::Viewport viewport {
            0.0f, 0.0f, static_cast<float>(m_extent.width), static_cast<float>(m_extent.height),
            0.0f, 1.0f};
        const vk::Rect2D scissor {{0, 0}, m_extent};
        command.setViewport(0, 1, &viewport);
        command.setScissor(0, 1, &scissor);
        command.draw(3, 1, 0, 0);
        command.endRendering();
}

vk::Rect2D Swapchain::ComputePresentRect(const vk::Extent2D& src) const {
        const auto&            win    = m_extent;
        const std::string&     aspect = Config::GetAspectRatio();
        const std::string&     filter = Config::GetUpscalingFilter();
        vk::Extent2D           size {win.width, win.height};
        if (filter == "integer") {
                // Largest integer multiple of the guest frame that fits in the window.
                const double sx    = static_cast<double>(win.width) / static_cast<double>(src.width);
                const double sy    = static_cast<double>(win.height) / static_cast<double>(src.height);
                const int    scale = static_cast<int>(std::floor(std::min(sx, sy)));
                if (scale >= 1) {
                        size = vk::Extent2D {src.width * static_cast<uint32_t>(scale),
                                             src.height * static_cast<uint32_t>(scale)};
                } else {
                        // Window smaller than the guest frame: fit with letterbox instead.
                        const double s = std::min(sx, sy);
                        size = vk::Extent2D {std::max(1u, static_cast<uint32_t>(static_cast<double>(src.width) * s)),
                                             std::max(1u, static_cast<uint32_t>(static_cast<double>(src.height) * s))};
                }
        } else {
                double ratio = 0.0;
                if (aspect == "16:9") {
                        ratio = 16.0 / 9.0;
                } else if (aspect == "16:10") {
                        ratio = 16.0 / 10.0;
                } else if (aspect == "4:3") {
                        ratio = 4.0 / 3.0;
                } else if (aspect == "21:9") {
                        ratio = 21.0 / 9.0;
                }
                if (ratio > 0.0) {
                        uint32_t height = win.height;
                        uint32_t width  = static_cast<uint32_t>(static_cast<double>(height) * ratio);
                        if (width > win.width) {
                                width  = win.width;
                                height = static_cast<uint32_t>(static_cast<double>(width) / ratio);
                        }
                        size = vk::Extent2D {width, height};
                }
        }
        const vk::Offset2D offset {static_cast<int32_t>((win.width - size.width) / 2u),
                                   static_cast<int32_t>((win.height - size.height) / 2u)};
        return vk::Rect2D {offset, size};
}

void Swapchain::DrawUpscale(vk::CommandBuffer command, Presenter::Frame& source,
                            const vk::Rect2D& rect) {
        auto& graphics = m_window.graphic_ctx;
        auto  device   = graphics.device;
        if (m_upscale_pipeline == nullptr) {
                vk::SamplerCreateInfo sampler {};
                sampler.magFilter    = vk::Filter::eLinear;
                sampler.minFilter    = vk::Filter::eLinear;
                sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
                sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
                sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
                RequireVulkanSuccess(device.createSampler(&sampler, nullptr, &m_upscale_sampler),
                                     "create upscale sampler");
                const vk::DescriptorSetLayoutBinding binding {
                    0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment};
                vk::DescriptorSetLayoutCreateInfo descriptors {};
                descriptors.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
                descriptors.bindingCount = 1;
                descriptors.pBindings    = &binding;
                RequireVulkanSuccess(
                    device.createDescriptorSetLayout(&descriptors, nullptr, &m_upscale_descriptors),
                    "create upscale descriptor layout");
                // FSR needs 5 x uvec4; the other filters read textureSize() and
                // simply ignore the constants.
                const vk::PushConstantRange constants {vk::ShaderStageFlagBits::eFragment, 0, 80u};
                vk::PipelineLayoutCreateInfo layout {};
                layout.setLayoutCount         = 1;
                layout.pSetLayouts            = &m_upscale_descriptors;
                layout.pushConstantRangeCount = 1;
                layout.pPushConstantRanges    = &constants;
                RequireVulkanSuccess(device.createPipelineLayout(&layout, nullptr, &m_upscale_layout),
                                     "create upscale pipeline layout");
                const auto vertex = CompileSPV(GPU_BLIT_FS_TRIANGLE_SPV, device);
                const std::string& filter = Config::GetUpscalingFilter();
                const auto fragment =
                    filter == "fsr"            ? CompileSPV(GPU_UPSCALE_FSR_SPV, device)
                    : filter == "scaleforce" ? CompileSPV(GPU_UPSCALE_SCALEFORCE_SPV, device)
                                             : CompileSPV(GPU_UPSCALE_BICUBIC_SPV, device);
                std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
                stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
                stages[0].module = vertex;
                stages[0].pName  = "main";
                stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
                stages[1].module = fragment;
                stages[1].pName  = "main";
                vk::PipelineVertexInputStateCreateInfo   vertex_input {};
                vk::PipelineInputAssemblyStateCreateInfo assembly {};
                assembly.topology = vk::PrimitiveTopology::eTriangleList;
                vk::PipelineViewportStateCreateInfo viewport {};
                viewport.viewportCount = 1;
                viewport.scissorCount  = 1;
                vk::PipelineRasterizationStateCreateInfo rasterization {};
                rasterization.lineWidth = 1.0f;
                vk::PipelineMultisampleStateCreateInfo multisample {};
                multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;
                vk::PipelineColorBlendAttachmentState attachment {};
                attachment.blendEnable    = VK_FALSE;
                attachment.colorWriteMask = vk::ColorComponentFlagBits::eR |
                                            vk::ColorComponentFlagBits::eG |
                                            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
                vk::PipelineColorBlendStateCreateInfo blend {};
                blend.attachmentCount = 1;
                blend.pAttachments    = &attachment;
                const std::array dynamic_states {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
                vk::PipelineDynamicStateCreateInfo dynamic {};
                dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
                dynamic.pDynamicStates    = dynamic_states.data();
                vk::PipelineRenderingCreateInfo rendering {};
                rendering.colorAttachmentCount    = 1;
                rendering.pColorAttachmentFormats = &m_format;
                vk::GraphicsPipelineCreateInfo create {};
                create.pNext               = &rendering;
                create.stageCount          = static_cast<uint32_t>(stages.size());
                create.pStages             = stages.data();
                create.pVertexInputState   = &vertex_input;
                create.pInputAssemblyState = &assembly;
                create.pViewportState      = &viewport;
                create.pRasterizationState = &rasterization;
                create.pMultisampleState   = &multisample;
                create.pColorBlendState    = &blend;
                create.pDynamicState       = &dynamic;
                create.layout              = m_upscale_layout;
                RequireVulkanSuccess(
                    device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &m_upscale_pipeline),
                    "create upscale pipeline");
                device.destroyShaderModule(fragment, nullptr);
                device.destroyShaderModule(vertex, nullptr);
        }
        const uint32_t sharpness = Config::GetUpscalingSharpness();
        if (sharpness > 0 && m_rcas_pipeline == nullptr) {
                // Second-pass FSR1 RCAS pipeline. Shares the upscale layout
                // (the RCAS shader reads only the first 32 push-constant
                // bytes), sampler and descriptor set layout.
                const auto vertex = CompileSPV(GPU_BLIT_FS_TRIANGLE_SPV, device);
                const auto fragment = CompileSPV(GPU_SHARPEN_RCAS_SPV, device);
                std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
                stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
                stages[0].module = vertex;
                stages[0].pName  = "main";
                stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
                stages[1].module = fragment;
                stages[1].pName  = "main";
                vk::PipelineVertexInputStateCreateInfo   vertex_input {};
                vk::PipelineInputAssemblyStateCreateInfo assembly {};
                assembly.topology = vk::PrimitiveTopology::eTriangleList;
                vk::PipelineViewportStateCreateInfo viewport {};
                viewport.viewportCount = 1;
                viewport.scissorCount  = 1;
                vk::PipelineRasterizationStateCreateInfo rasterization {};
                rasterization.lineWidth = 1.0f;
                vk::PipelineMultisampleStateCreateInfo multisample {};
                multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;
                vk::PipelineColorBlendAttachmentState attachment {};
                attachment.blendEnable    = VK_FALSE;
                attachment.colorWriteMask = vk::ColorComponentFlagBits::eR |
                                            vk::ColorComponentFlagBits::eG |
                                            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
                vk::PipelineColorBlendStateCreateInfo blend {};
                blend.attachmentCount = 1;
                blend.pAttachments    = &attachment;
                const std::array dynamic_states {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
                vk::PipelineDynamicStateCreateInfo dynamic {};
                dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
                dynamic.pDynamicStates    = dynamic_states.data();
                vk::PipelineRenderingCreateInfo rendering {};
                rendering.colorAttachmentCount    = 1;
                rendering.pColorAttachmentFormats = &m_format;
                vk::GraphicsPipelineCreateInfo create {};
                create.pNext               = &rendering;
                create.stageCount          = static_cast<uint32_t>(stages.size());
                create.pStages             = stages.data();
                create.pVertexInputState   = &vertex_input;
                create.pInputAssemblyState = &assembly;
                create.pViewportState      = &viewport;
                create.pRasterizationState = &rasterization;
                create.pMultisampleState   = &multisample;
                create.pColorBlendState    = &blend;
                create.pDynamicState       = &dynamic;
                create.layout              = m_upscale_layout;
                RequireVulkanSuccess(
                    device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &m_rcas_pipeline),
                    "create RCAS sharpening pipeline");
                device.destroyShaderModule(fragment, nullptr);
                device.destroyShaderModule(vertex, nullptr);
        }
        if (source.view == nullptr) {
                vk::ImageViewCreateInfo view {};
                view.image            = source.image.image;
                view.viewType         = vk::ImageViewType::e2D;
                view.format           = source.image.format;
                view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
                RequireVulkanSuccess(device.createImageView(&view, nullptr, &source.view),
                                     "create upscale image view");
        }
        const vk::DescriptorImageInfo image {m_upscale_sampler, source.view,
                                             vk::ImageLayout::eShaderReadOnlyOptimal};
        vk::WriteDescriptorSet        write {};
        write.dstBinding      = 0;
        write.descriptorCount = 1;
        write.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
        write.pImageInfo      = &image;
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, m_upscale_layout, 0, 1, &write);

        // AMD FsrEasuCon() constants; bicubic and ScaleForce ignore them and
        // take sizes from textureSize().
        std::array<uint32_t, 20> constants {};
        const float in_w       = static_cast<float>(source.image.extent.width);
        const float in_h       = static_cast<float>(source.image.extent.height);
        const float out_w      = static_cast<float>(rect.extent.width);
        const float out_h      = static_cast<float>(rect.extent.height);
        const auto  as_uint    = [](float value) {
                uint32_t bits = 0;
                static_assert(sizeof(float) == sizeof(bits));
                std::memcpy(&bits, &value, sizeof(bits));
                return bits;
        };
        // Output integer position to a pixel position in viewport.
        constants[0]  = as_uint(in_w / out_w);
        constants[1]  = as_uint(in_h / out_h);
        constants[2]  = as_uint(0.5f * in_w / out_w - 0.5f);
        constants[3]  = as_uint(0.5f * in_h / out_h - 0.5f);
        // Viewport pixel position to normalized image space + gather offsets.
        constants[4]  = as_uint(1.0f / in_w);
        constants[5]  = as_uint(1.0f / in_h);
        constants[6]  = as_uint(1.0f / in_w);
        constants[7]  = as_uint(-1.0f / in_h);
        constants[8]  = as_uint(-1.0f / in_w);
        constants[9]  = as_uint(2.0f / in_h);
        constants[10] = as_uint(1.0f / in_w);
        constants[11] = as_uint(2.0f / in_h);
        constants[12] = as_uint(0.0f);
        constants[13] = as_uint(4.0f / in_h);
        constants[16] = rect.extent.width;
        constants[17] = rect.extent.height;
        command.pushConstants(m_upscale_layout, vk::ShaderStageFlagBits::eFragment, 0,
                              sizeof(constants), constants.data());
        command.bindPipeline(vk::PipelineBindPoint::eGraphics, m_upscale_pipeline);
        const vk::Viewport viewport {
            static_cast<float>(rect.offset.x), static_cast<float>(rect.offset.y),
            static_cast<float>(rect.extent.width), static_cast<float>(rect.extent.height), 0.0f, 1.0f};
        const vk::Rect2D scissor {{0, 0}, m_extent};

        // FSR1 RCAS sharpening renders the upscaling filter into an
        // intermediate target first, then sharpens it into the swapchain as a
        // second pass. Falls back to the single-pass path when the format
        // lacks sampled color attachment support or creation failed.
        if (sharpness > 0 && EnsureSharpenTarget()) {
                auto sharpen_barrier = [&](vk::PipelineStageFlags2 dst_stage,
                                           vk::AccessFlags2          dst_access,
                                           vk::ImageLayout           dst_layout) {
                        vk::ImageMemoryBarrier2 barrier {};
                        barrier.srcStageMask        = m_sharpen_stage;
                        barrier.srcAccessMask       = m_sharpen_access;
                        barrier.dstStageMask        = dst_stage;
                        barrier.dstAccessMask       = dst_access;
                        barrier.oldLayout           = m_sharpen_layout;
                        barrier.newLayout           = dst_layout;
                        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        barrier.image               = m_sharpen_image.image;
                        barrier.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
                        vk::DependencyInfo dependency {};
                        dependency.imageMemoryBarrierCount = 1;
                        dependency.pImageMemoryBarriers    = &barrier;
                        command.pipelineBarrier2(dependency);
                        m_sharpen_stage  = dst_stage;
                        m_sharpen_access = dst_access;
                        m_sharpen_layout = dst_layout;
                };

                // Pass 1: upscaling filter into the intermediate target.
                sharpen_barrier(vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                vk::AccessFlagBits2::eColorAttachmentWrite,
                                vk::ImageLayout::eColorAttachmentOptimal);
                vk::RenderingAttachmentInfo intermediate_attachment {};
                intermediate_attachment.imageView      = m_sharpen_view;
                intermediate_attachment.imageLayout    = vk::ImageLayout::eColorAttachmentOptimal;
                intermediate_attachment.loadOp         = vk::AttachmentLoadOp::eDontCare;
                intermediate_attachment.storeOp        = vk::AttachmentStoreOp::eStore;
                vk::RenderingInfo intermediate_rendering {};
                intermediate_rendering.renderArea.extent    = m_extent;
                intermediate_rendering.layerCount           = 1;
                intermediate_rendering.colorAttachmentCount = 1;
                intermediate_rendering.pColorAttachments    = &intermediate_attachment;
                command.beginRendering(&intermediate_rendering);
                command.setViewport(0, 1, &viewport);
                command.setScissor(0, 1, &scissor);
                command.draw(3, 1, 0, 0);
                command.endRendering();

                // Pass 2: RCAS sharpening into the swapchain image.
                sharpen_barrier(vk::PipelineStageFlagBits2::eFragmentShader,
                                vk::AccessFlagBits2::eShaderRead,
                                vk::ImageLayout::eShaderReadOnlyOptimal);
                std::array<uint32_t, 20> rcas_constants {};
                rcas_constants[0]       = std::clamp(sharpness, 0u, 100u) * 255u / 100u;
                rcas_constants[4]       = static_cast<uint32_t>(rect.offset.x);
                rcas_constants[5]       = static_cast<uint32_t>(rect.offset.y);
                rcas_constants[6]       = rect.extent.width;
                rcas_constants[7]       = rect.extent.height;
                const vk::DescriptorImageInfo sharpened_image {
                    m_upscale_sampler, m_sharpen_view, vk::ImageLayout::eShaderReadOnlyOptimal};
                vk::WriteDescriptorSet sharpened_write {};
                sharpened_write.dstBinding      = 0;
                sharpened_write.descriptorCount = 1;
                sharpened_write.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
                sharpened_write.pImageInfo      = &sharpened_image;
                command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, m_upscale_layout, 0, 1,
                                             &sharpened_write);
                command.pushConstants(m_upscale_layout, vk::ShaderStageFlagBits::eFragment, 0,
                                      sizeof(rcas_constants), rcas_constants.data());
                command.bindPipeline(vk::PipelineBindPoint::eGraphics, m_rcas_pipeline);
                vk::RenderingAttachmentInfo attachment {};
                attachment.imageView        = m_image_views[m_image_index];
                attachment.imageLayout      = vk::ImageLayout::eColorAttachmentOptimal;
                attachment.loadOp           = vk::AttachmentLoadOp::eClear;
                attachment.storeOp          = vk::AttachmentStoreOp::eStore;
                attachment.clearValue.color = vk::ClearColorValue {std::array<float, 4> {0.0f, 0.0f, 0.0f, 1.0f}};
                vk::RenderingInfo rendering {};
                rendering.renderArea.extent    = m_extent;
                rendering.layerCount           = 1;
                rendering.colorAttachmentCount = 1;
                rendering.pColorAttachments    = &attachment;
                command.beginRendering(&rendering);
                command.setViewport(0, 1, &viewport);
                command.setScissor(0, 1, &scissor);
                command.draw(3, 1, 0, 0);
                command.endRendering();
                return;
        }

        vk::RenderingAttachmentInfo attachment {};
        attachment.imageView        = m_image_views[m_image_index];
        attachment.imageLayout      = vk::ImageLayout::eColorAttachmentOptimal;
        attachment.loadOp           = vk::AttachmentLoadOp::eClear;
        attachment.storeOp          = vk::AttachmentStoreOp::eStore;
        attachment.clearValue.color = vk::ClearColorValue {std::array<float, 4> {0.0f, 0.0f, 0.0f, 1.0f}};
        vk::RenderingInfo rendering {};
        rendering.renderArea.extent    = m_extent;
        rendering.layerCount           = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments    = &attachment;
        command.beginRendering(&rendering);
        command.setViewport(0, 1, &viewport);
        command.setScissor(0, 1, &scissor);
        command.draw(3, 1, 0, 0);
        command.endRendering();
}

// Lazily allocate (or reallocate after a swapchain resize) the intermediate
// color target consumed by the RCAS sharpening pass. Returns false when the
// swapchain format cannot be rendered to and sampled from, in which case the
// caller falls back to the single-pass present path.
bool Swapchain::EnsureSharpenTarget() {
        auto& graphics = m_window.graphic_ctx;
        if (m_sharpen_image.image != nullptr && m_sharpen_view != nullptr &&
            m_sharpen_image.extent.width == m_extent.width &&
            m_sharpen_image.extent.height == m_extent.height && m_sharpen_image.format == m_format) {
                return true;
        }
        if (m_sharpen_view != nullptr) {
                graphics.device.destroyImageView(m_sharpen_view, nullptr);
                m_sharpen_view = nullptr;
        }
        if (m_sharpen_image.image != nullptr) {
                graphics.DeleteImage(m_sharpen_image);
                m_sharpen_image.image = nullptr;
        }
        m_sharpen_stage  = vk::PipelineStageFlagBits2::eTopOfPipe;
        m_sharpen_access = vk::AccessFlagBits2::eNone;
        m_sharpen_layout = vk::ImageLayout::eUndefined;

        const auto features = graphics.GetFormatProperties(m_format).optimalTilingFeatures;
        const auto required = vk::FormatFeatureFlagBits::eColorAttachment |
                              vk::FormatFeatureFlagBits::eSampled;
        if ((features & required) != required) {
                static std::atomic_bool warned {false};
                if (!warned.exchange(true)) {
                        LOGF("RCAS sharpening disabled: swapchain format %d lacks sampled color "
                             "attachment support\n",
                             static_cast<int>(m_format));
                }
                return false;
        }

        vk::ImageCreateInfo create {};
        create.imageType     = vk::ImageType::e2D;
        create.extent        = {m_extent.width, m_extent.height, 1};
        create.mipLevels     = 1;
        create.arrayLayers   = 1;
        create.format        = m_format;
        create.tiling        = vk::ImageTiling::eOptimal;
        create.initialLayout = vk::ImageLayout::eUndefined;
        create.usage         = vk::ImageUsageFlagBits::eColorAttachment |
                       vk::ImageUsageFlagBits::eSampled;
        create.sharingMode = vk::SharingMode::eExclusive;
        create.samples     = vk::SampleCountFlagBits::e1;
        if (!graphics.CreateImage(create, m_sharpen_image)) {
                LOGF("failed to allocate the RCAS sharpening intermediate target\n");
                return false;
        }
        vk::ImageViewCreateInfo view {};
        view.image            = m_sharpen_image.image;
        view.viewType         = vk::ImageViewType::e2D;
        view.format           = m_format;
        view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        RequireVulkanSuccess(graphics.device.createImageView(&view, nullptr, &m_sharpen_view),
                             "create RCAS sharpening intermediate view");
        return true;
}

void Swapchain::RecordPresentCommands(CommandBuffer& command, Presenter::Frame* source,
                                      const Presenter::Layer& overlay, bool draw_system_overlay) {
        EXIT_IF(m_image_index >= m_images.size());
        auto       vk_command      = command.Handle();
        const bool draw_overlay    = overlay.frame != nullptr;
        const bool draw_attachment = draw_overlay || draw_system_overlay;
        const std::string& filter  = Config::GetUpscalingFilter();
        const bool upscale_shader =
            source != nullptr &&
            (filter == "fsr" || filter == "scaleforce" || filter == "bicubic");
        m_present_uses_graphics = upscale_shader;
        const auto present_rect =
            source != nullptr
                ? ComputePresentRect(vk::Extent2D {source->image.extent.width, source->image.extent.height})
                : vk::Rect2D {{0, 0}, m_extent};
        if (source != nullptr) {
                source->Transit(vk_command,
                                upscale_shader ? vk::ImageLayout::eShaderReadOnlyOptimal
                                               : vk::ImageLayout::eTransferSrcOptimal,
                                upscale_shader ? vk::AccessFlagBits2::eShaderRead
                                               : vk::AccessFlagBits2::eTransferRead);
        }
        if (draw_overlay) {
                overlay.frame->Transit(vk_command, vk::ImageLayout::eShaderReadOnlyOptimal,
                                       vk::AccessFlagBits2::eShaderRead);
        }

        vk::ImageMemoryBarrier to_transfer {};
        to_transfer.sType                           = vk::StructureType::eImageMemoryBarrier;
        to_transfer.dstAccessMask = upscale_shader ? vk::AccessFlagBits::eColorAttachmentWrite
                                                   : vk::AccessFlagBits::eTransferWrite;
        to_transfer.oldLayout                       = vk::ImageLayout::eUndefined;
        to_transfer.newLayout = upscale_shader ? vk::ImageLayout::eColorAttachmentOptimal
                                               : vk::ImageLayout::eTransferDstOptimal;
        to_transfer.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        to_transfer.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        to_transfer.image                           = m_images[m_image_index];
        to_transfer.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
        to_transfer.subresourceRange.baseMipLevel   = 0;
        to_transfer.subresourceRange.levelCount     = 1;
        to_transfer.subresourceRange.baseArrayLayer = 0;
        to_transfer.subresourceRange.layerCount     = 1;
        // Match the acquire wait stage so the layout transition cannot precede acquisition.
        vk_command.pipelineBarrier(upscale_shader ? vk::PipelineStageFlagBits::eColorAttachmentOutput
                                                  : vk::PipelineStageFlagBits::eTransfer,
                                   upscale_shader ? vk::PipelineStageFlagBits::eColorAttachmentOutput
                                                  : vk::PipelineStageFlagBits::eTransfer,
                                   vk::DependencyFlags {}, 0, nullptr, 0, nullptr, 1, &to_transfer);

        if (upscale_shader) {
                DrawUpscale(vk_command, *source, present_rect);
        } else if (source != nullptr) {
                const bool letterboxed = present_rect.offset.x != 0 || present_rect.offset.y != 0 ||
                                         present_rect.extent.width != m_extent.width ||
                                         present_rect.extent.height != m_extent.height;
                if (letterboxed) {
                        const vk::ClearColorValue black {};
                        vk_command.clearColorImage(m_images[m_image_index],
                                                   vk::ImageLayout::eTransferDstOptimal, &black, 1,
                                                   &to_transfer.subresourceRange);
                }
                vk::ImageBlit region {};
                region.srcSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
                region.srcSubresource.mipLevel       = 0;
                region.srcSubresource.baseArrayLayer = 0;
                region.srcSubresource.layerCount     = 1;
                region.srcOffsets[1].x               = static_cast<int>(source->image.extent.width);
                region.srcOffsets[1].y               = static_cast<int>(source->image.extent.height);
                region.srcOffsets[1].z               = 1;
                region.dstSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
                region.dstSubresource.mipLevel       = 0;
                region.dstSubresource.baseArrayLayer = 0;
                region.dstSubresource.layerCount     = 1;
                region.dstOffsets[0].x               = present_rect.offset.x;
                region.dstOffsets[0].y               = present_rect.offset.y;
                region.dstOffsets[1].x               =
                    present_rect.offset.x + static_cast<int>(present_rect.extent.width);
                region.dstOffsets[1].y =
                    present_rect.offset.y + static_cast<int>(present_rect.extent.height);
                region.dstOffsets[1].z = 1;
                vk_command.blitImage(source->image.image, vk::ImageLayout::eTransferSrcOptimal,
                                     m_images[m_image_index], vk::ImageLayout::eTransferDstOptimal, 1,
                                     &region,
                                     filter == "nearest" ? vk::Filter::eNearest : vk::Filter::eLinear);
        } else {
                const vk::ClearColorValue black {};
                vk_command.clearColorImage(m_images[m_image_index], vk::ImageLayout::eTransferDstOptimal,
                                           &black, 1, &to_transfer.subresourceRange);
        }

        vk::ImageMemoryBarrier to_present {};
        to_present.sType = vk::StructureType::eImageMemoryBarrier;
        to_present.srcAccessMask = upscale_shader ? vk::AccessFlagBits::eColorAttachmentWrite
                                                  : vk::AccessFlagBits::eTransferWrite;
        to_present.dstAccessMask       = draw_attachment ? vk::AccessFlagBits::eColorAttachmentRead |
                                                               vk::AccessFlagBits::eColorAttachmentWrite
                                                         : vk::AccessFlagBits::eMemoryRead;
        to_present.oldLayout = upscale_shader ? vk::ImageLayout::eColorAttachmentOptimal
                                              : vk::ImageLayout::eTransferDstOptimal;
        to_present.newLayout           = draw_attachment ? vk::ImageLayout::eColorAttachmentOptimal
                                                         : vk::ImageLayout::ePresentSrcKHR;
        to_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_present.image               = m_images[m_image_index];
        to_present.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
        to_present.subresourceRange.baseMipLevel   = 0;
        to_present.subresourceRange.levelCount     = 1;
        to_present.subresourceRange.baseArrayLayer = 0;
        to_present.subresourceRange.layerCount     = 1;
        vk_command.pipelineBarrier(upscale_shader ? vk::PipelineStageFlagBits::eColorAttachmentOutput
                                                  : vk::PipelineStageFlagBits::eTransfer,
                                   draw_attachment ? vk::PipelineStageFlagBits::eColorAttachmentOutput
                                                   : vk::PipelineStageFlagBits::eAllCommands,
                                   vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1,
                                   &to_present);
        if (draw_attachment) {
                if (draw_overlay) {
                        DrawOverlay(vk_command, overlay);
                        if (draw_system_overlay) {
                                to_present.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
                                to_present.oldLayout     = vk::ImageLayout::eColorAttachmentOptimal;
                                vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                                           vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0,
                                                           nullptr, 1, &to_present);
                        }
                }
                if (draw_system_overlay) {
                        m_system_overlay->Record(vk_command, m_image_views[m_image_index]);
                }
                to_present.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
                to_present.dstAccessMask = vk::AccessFlagBits::eMemoryRead;
                to_present.oldLayout     = vk::ImageLayout::eColorAttachmentOptimal;
                to_present.newLayout     = vk::ImageLayout::ePresentSrcKHR;
                vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                           vk::PipelineStageFlagBits::eAllCommands,
                                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1,
                                           &to_present);
        }
}

uint64_t Swapchain::Submit(CommandScheduler& scheduler) {
        EXIT_IF(m_frame_index >= m_image_acquired.size() || m_image_index >= m_render_complete.size());
        SubmitInfo submit;
        submit.AddWait(m_image_acquired[m_frame_index], 1,
                       m_present_uses_graphics ? vk::PipelineStageFlagBits::eColorAttachmentOutput
                                               : vk::PipelineStageFlagBits::eTransfer);
        submit.AddSignal(m_render_complete[m_image_index]);
        return scheduler.Submit(submit);
}

Swapchain::Status Swapchain::Present() {
        EXIT_IF(m_image_index >= m_render_complete.size());
        const auto         ready = m_render_complete[m_image_index];
        vk::PresentInfoKHR present {};
        present.sType              = vk::StructureType::ePresentInfoKHR;
        present.swapchainCount     = 1;
        present.pSwapchains        = &m_handle;
        present.pImageIndices      = &m_image_index;
        present.pWaitSemaphores    = &ready;
        present.waitSemaphoreCount = 1;

        vk::Result result;
        {
                Common::LockGuard lock(m_window.graphic_ctx.queue_mutex);
                result = m_window.graphic_ctx.queue.presentKHR(&present);
        }
        switch (result) {
                case vk::Result::eSuccess: break;
                case vk::Result::eSuboptimalKHR:
                        LOGF("vkQueuePresentKHR returned vk::Result::eSuboptimalKHR\n");
                        return Status::Recreate;
                case vk::Result::eErrorOutOfDateKHR:
                        LOGF("vkQueuePresentKHR returned vk::Result::eErrorOutOfDateKHR\n");
                        return Status::Recreate;
                case vk::Result::eErrorSurfaceLostKHR:
                        LOGF("vkQueuePresentKHR returned vk::Result::eErrorSurfaceLostKHR\n");
                        return Status::SurfaceLost;
                default: EXIT("vkQueuePresentKHR failed: %s\n", vk::to_string(result).c_str());
        }
        m_frame_index = (m_frame_index + 1u) % static_cast<uint32_t>(m_images.size());
        return Status::Success;
}

Presenter::Presenter(WindowContext& window): m_impl(std::make_unique<Impl>(window)) {}

Presenter::~Presenter() = default;

Presenter::Frame& Presenter::PrepareFrame(CommandBuffer& buffer, const ImageInfo& info, bool hdr) {
        KYTY_PROFILER_FUNCTION();
        EXIT_IF(buffer.IsInvalid());
        auto frame_format = info.pixel_format;
        switch (frame_format) {
                case vk::Format::eR8G8B8A8Srgb: frame_format = vk::Format::eR8G8B8A8Unorm; break;
                case vk::Format::eB8G8R8A8Srgb: frame_format = vk::Format::eB8G8R8A8Unorm; break;
                default: break;
        }
        auto* frame = m_impl->frames.Acquire({info.extent.width, info.extent.height}, frame_format);
        frame->hdr   = hdr;
        frame->blank = false;
        Common::LockGuard render_lock(m_impl->renderer.GetMutex());
        auto&             image = m_impl->ResolveSurface(info);
        if (image.backing.format == vk::Format::eUndefined) {
                EXIT("unsupported presentation source, image=%p\n", static_cast<const void*>(&image));
        }
        frame->Configure(m_impl->window.graphic_ctx,
                         {image.backing.extent.width, image.backing.extent.height}, frame_format);
        frame->CopyFrom(buffer, image);
        return *frame;
}

Presenter::Frame& Presenter::PrepareBlankFrame(uint32_t width, uint32_t height, bool opaque,
                                               CommandBuffer* producer) {
        KYTY_PROFILER_FUNCTION();
        auto              format = m_impl->frames.GetFormat();
        auto*             frame  = m_impl->frames.Acquire({width, height}, format);
        frame->hdr   = false;
        frame->blank = true;
        Common::LockGuard render_lock(m_impl->renderer.GetMutex());
        frame->Configure(m_impl->window.graphic_ctx, {width, height}, format);
        vk::ClearColorValue clear {};
        clear.float32[3] = opaque ? 1.0f : 0.0f;
        if (producer != nullptr) {
                EXIT_IF(producer->IsInvalid());
                frame->Clear(*producer, clear);
        } else {
                auto& command = m_impl->present_scheduler.BeginCommand();
                frame->Clear(command, clear);
                frame->present_tick = m_impl->present_scheduler.Submit();
        }
        return *frame;
}

bool Presenter::PresentLastFrame() {
        Common::LockGuard lock(m_impl->present_mutex);
        if (m_impl->layers[0].frame == nullptr && m_impl->layers[1].frame == nullptr) {
                return false;
        }
        m_impl->Present();
        return true;
}

bool Presenter::IsGuestPaused() const noexcept {
        return m_impl->window.loop.paused.load(std::memory_order_acquire);
}

bool Presenter::IsHdrSupported() const noexcept {
        return m_impl->swapchain.IsHdrSupported();
}

bool Presenter::NeedsSystemOverlayRefresh() const noexcept {
        const auto visual = GetSystemOverlayVisualState();
        return visual.active ||
               visual.revision != m_impl->presented_overlay_revision.load(std::memory_order_acquire);
}

RenderContext& Presenter::Renderer() const noexcept {
        return m_impl->renderer;
}

void Presenter::Present(Frame& frame) {
        const Layer layer {&frame, 0, false};
        Present(std::span(&layer, 1));
}

void Presenter::Present(std::span<const Layer> layers) {
        Common::LockGuard lock(m_impl->present_mutex);
        for (const auto& layer: layers) {
                EXIT_IF(layer.bus < 0 || layer.bus >= static_cast<int>(m_impl->layers.size()));
                m_impl->frames.ValidateForPresent(layer.frame);
                auto& previous = m_impl->layers[layer.bus];
                if (previous.frame != nullptr) {
                        m_impl->frames.Release(previous.frame);
                }
                previous = layer;
        }
        m_impl->Present();
}

void Presenter::ClearLayer(int bus) {
        if (static_cast<size_t>(bus) >= m_impl->layers.size()) {
                return;
        }
        Common::LockGuard lock(m_impl->present_mutex);
        auto& layer = m_impl->layers[bus];
        if (layer.frame != nullptr) {
                m_impl->frames.Release(layer.frame);
                layer = {};
        }
}

void Presenter::Impl::Present() {
        KYTY_PROFILER_FUNCTION();

        const auto overlay_visual = GetSystemOverlayVisualState();
        // Keep black/blank flips in the current color space; only real main-bus frames switch it.
        const auto* main_frame = layers[0].frame;
        const bool hdr_changed = main_frame != nullptr && !main_frame->blank &&
                                 swapchain.SetHdr(main_frame->hdr);
        // Some window systems keep presenting an old swapchain after a resize.
        if (hdr_changed || swapchain.NeedsResize()) {
                RecoverSwapchain(Swapchain::Status::Recreate);
        }
        for (uint32_t attempt = 0; attempt < 2; attempt++) {
                auto status = swapchain.AcquireNextImage();
                if (status != Swapchain::Status::Success) {
                        RecoverSwapchain(status);
                        continue;
                }
                {
                        Common::LockGuard render_lock(renderer.GetMutex());
                        auto&             command = present_scheduler.BeginCommand();
                        const bool        draw_system_overlay =
                            overlay_visual.active && swapchain.PrepareSystemOverlay();
                        swapchain.RecordPresentCommands(command, layers[0].frame, layers[1],
                                                        draw_system_overlay);
                        const auto tick = swapchain.Submit(present_scheduler);
                        for (const auto& layer: layers) {
                                if (layer.frame != nullptr) {
                                        layer.frame->present_tick = tick;
                                }
                        }
                }
                status = swapchain.Present();
                if (status != Swapchain::Status::Success) {
                        RecoverSwapchain(status);
                        continue;
                }

                presented_overlay_revision.store(overlay_visual.revision, std::memory_order_release);
                window.loop.presented_frames.fetch_add(1, std::memory_order_relaxed);
                return;
        }
        LOGF("Vulkan presentation retry exhausted; dropping frame\n");
}

void Presenter::Discard(Frame& frame) {
        m_impl->frames.Release(&frame);
}

WindowContext::WindowContext() = default;

} // namespace Libs::Graphics
