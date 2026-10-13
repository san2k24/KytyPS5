#ifndef KYTY_COMMON_EMULATOR_CONFIG_H_
#define KYTY_COMMON_EMULATOR_CONFIG_H_

#include "common/common.h"

#include <cstddef>
#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace Config {

void Initialize();
void Shutdown();

struct Lifecycle {
        static constexpr const char* name       = "Config";
        static constexpr auto        initialize = Config::Initialize;
        static constexpr auto        shutdown   = Config::Shutdown;
};

enum class ShaderOptimizationType { None, Size, Performance };

enum class LogDirection { Silent, Console, File };

enum class PresentMode { Fifo, Mailbox, Immediate };

using Keymap = std::vector<std::string>;
using ControllerColor = std::array<uint8_t, 3>;

constexpr uint32_t DEFAULT_CONSOLE_LANGUAGE = 1;
constexpr uint32_t MAX_CONSOLE_LANGUAGE     = 29;
constexpr std::size_t MAX_USER_NAME_LENGTH = 16;
constexpr int32_t DEFAULT_USER_ID           = 1000;

constexpr bool IsConfiguredUserIdValid(int32_t user_id) {
        constexpr int32_t USER_ID_EVERYONE = 0xfe;
        constexpr int32_t USER_ID_SYSTEM   = 0xff;
        return user_id >= 0 && user_id != USER_ID_EVERYONE && user_id != USER_ID_SYSTEM;
}

struct ConfigOptions {
        uint32_t               screen_width                = 1280;
        uint32_t               screen_height               = 720;
        std::string            user_name                   = "Kyty";
        int32_t                user_id                     = DEFAULT_USER_ID;
        std::string            audio_input_device;
        std::optional<ControllerColor> controller_color;
        uint32_t               controller_speaker_volume      = 50;
        uint32_t               controller_vibration_intensity = 100;
        PresentMode            present_mode                = PresentMode::Mailbox;
        int32_t                gpu_index                   = -1;
        bool                   fullscreen_enabled          = false;
        bool                   hdr_enabled                 = false;
        bool                   hide_cursor_enabled         = false;
        bool                   splash_enabled              = true;
        bool                   vr_enabled                  = false;
        bool                   amd_cpu_enabled             = false;
        uint32_t               vblank_frequency            = 60;
        // When true, the present thread fires additional vblank events to keep
        // game logic at the configured vblank_frequency even if the host GPU
        // cannot render at that rate. This prevents the 'slow motion below 60
        // FPS' behavior at the cost of dropped rendered frames (visuals become
        // choppy but the game runs at the correct speed). Default ON for forks
        // targeting GPUs below PS5 spec (e.g. AMD RDNA1 / Radeon RX 5500 XT).
        bool                   vblank_catchup_enabled       = true;
        // Upscaling filter applied when the swapchain is bigger than the guest
        // render target. Accepted values: "nearest" (sharp pixels), "bilinear"
        // (smooth), "bicubic" (sharper smooth), "integer" (pixel-perfect,
        // only when host resolution is an integer multiple of guest).
        std::string            upscaling_filter             = "bilinear";
        // FSR1 RCAS sharpening strength applied after the upscaling filter
        // (fsr/scaleforce/bicubic only). 0 disables the RCAS pass entirely;
        // 100 is the maximum sharpness. Default 80 approximates the AMD
        // FidelityFX recommended default.
        uint32_t               upscaling_sharpness          = 80;
        // Texture filtering override for guest samplers: "game" keeps the
        // exact sampler state programmed by the game (faithful), "force_af"
        // forces 16x anisotropic filtering on non-compare samplers (crisper
        // textures at oblique angles, visual-only change).
        std::string            texture_filtering            = "game";
        // PPSSPP-style 2x speed toggle: when true, the present thread fires 2
        // vblanks for every rendered frame, making the game logic advance at
        // 2x the configured vblank frequency (i.e. 120 Hz with the default 60
        // Hz). Useful for fast-forwarding cutscenes or replaying content. The
        // game runs at 2x speed (audio pitched up, physics faster). Hold a hotkey
        // or set this in the launcher to enable.
        bool                   speed_2x_enabled             = false;
        // PPSSPP-style frame skip: when the host GPU is behind by N vblank
        // periods, skip rendering for N-1 frames and just fire vblank events
        // for game logic. Different from vblank_catchup: catch-up fires ONLY
        // the missed vblanks; frame_skip N fires N vblanks per loop iteration
        // and renders only the last. With frame_skip=1 (default) the behavior
        // matches the existing catch-up. Higher values trade visual smoothness
        // for game-logic speed stability on very slow GPUs.
        uint32_t                frame_skip_count             = 1;
        // How the guest (typically 16:9) frame maps onto the host window:
        // "stretch" fills the whole window (legacy behavior), any other value
        // letterboxes the frame inside the window keeping the given ratio.
        // Accepted values: "stretch", "16:9", "16:10", "4:3", "21:9".
        std::string            aspect_ratio                 = "stretch";
        // Number of swapchain images requested from the host compositor.
        // 0 selects the automatic behavior (minImageCount + 1, usually 3).
        // 2 trades throughput for the lowest present latency, 3 is the
        // balanced default, 4 adds buffering for smoother pacing on slow GPUs.
        uint32_t               max_queued_frames            = 0;
        // SDL playback device used for the main audio output. Empty selects
        // the system default device.
        std::string            audio_output_device;
        // Master volume percentage applied to the main audio output only
        // (controller speaker volume has its own slider). 100 = untouched.
        uint32_t               master_volume                = 100;
        // When true, the Vulkan driver pipeline cache is loaded from and saved
        // to _PipelineCache/<title id>.bin, removing most first-run shader
        // compilation stutter. Disabling it forces recompilation every run
        // (useful to reproduce shader-related bugs).
        bool                   shader_cache_enabled         = true;
        // When true (and the shader cache is enabled), the pipeline cache is
        // also saved periodically during gameplay from a background thread and
        // on every crash-safe checkpoint, so warm runs survive emulated crashes.
        bool                   shader_cache_async_save      = true;
        // Host CPU cores the emulator may run on. Empty means no restriction.
        // Accepted forms: "0-7", "0,2,4", "0-3,8,9" (same core list as
        // Windows processor affinity masks; 0 <= core < 64).
        std::string            cpu_affinity;
        uint32_t               console_language            = DEFAULT_CONSOLE_LANGUAGE;
        bool                   vulkan_validation_enabled   = false;
        bool                   shader_validation_enabled   = false;
        ShaderOptimizationType shader_optimization_type    = ShaderOptimizationType::None;
        LogDirection           shader_log_direction        = LogDirection::Silent;
        std::filesystem::path  shader_log_folder           = "_Shaders";
        bool                   command_buffer_dump_enabled = false;
        std::filesystem::path  command_buffer_dump_folder  = "_Buffers";
        bool                   graphics_debug_dump_enabled = false;
        LogDirection           printf_direction            = LogDirection::Silent;
        std::filesystem::path  printf_output_file          = "_kyty.txt";
        bool                   profiler_enabled            = false;
        bool                   spirv_debug_printf_enabled  = false;
        bool                   gpu_assisted_validation_enabled = false;
        bool                   renderdoc_enabled           = false;
        bool                   readback_linear_images      = false;
        bool                   sync_raw_image_buffers      = false;
        bool                   tessellation_enabled        = false;
        bool                   trophy_enabled              = true;
        bool                   playgo_hack_enabled         = false;
        bool                   skip_notice_screen          = false;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
        bool red_zone_protection_enabled = false;
#endif
        Keymap keymap;
        // Ordinary boots need a non-null argv[0]; --guest-exec supplies the caller's arguments.
        std::vector<std::string> guest_arguments {""};
};

void Load(const ConfigOptions& cfg);

uint32_t GetScreenWidth();
uint32_t GetScreenHeight();
const std::string& GetUserName();
int32_t  GetUserId();
const std::string& GetAudioInputDevice();
const std::optional<ControllerColor>& GetControllerColor();
uint32_t GetControllerSpeakerVolume();
uint32_t GetControllerVibrationIntensity();
PresentMode GetPresentMode();
int32_t GetGpuIndex();
bool     FullscreenEnabled();
bool     HdrEnabled();
bool     HideCursorEnabled();
bool     VrEnabled();
bool     AmdCpuEnabled();
uint32_t GetVblankFrequency();
bool     VblankCatchupEnabled();
const std::string& GetUpscalingFilter();
uint32_t GetUpscalingSharpness();
const std::string& GetTextureFiltering();
bool     Speed2xEnabled();
uint32_t GetFrameSkipCount();
const std::string& GetAspectRatio();
uint32_t GetMaxQueuedFrames();
const std::string& GetAudioOutputDevice();
uint32_t GetMasterVolume();
bool     ShaderCacheEnabled();
bool     ShaderCacheAsyncSaveEnabled();
const std::string& GetCpuAffinity();
uint32_t GetConsoleLanguage();
bool     VulkanValidationEnabled();

bool                   ShaderValidationEnabled();
ShaderOptimizationType GetShaderOptimizationType();
LogDirection           GetShaderLogDirection();
std::filesystem::path  GetShaderLogFolder();

bool                  CommandBufferDumpEnabled();
std::filesystem::path GetCommandBufferDumpFolder();

bool GraphicsDebugDumpEnabled();

LogDirection          GetPrintfDirection();
std::filesystem::path GetPrintfOutputFile();

bool ProfilerEnabled();

bool SpirvDebugPrintfEnabled();

bool GpuAssistedValidationEnabled();

bool RenderDocEnabled();
bool ReadbackLinearImagesEnabled();
bool SyncRawImageBuffersEnabled();
bool TessellationEnabled();
bool TrophyEnabled();
bool PlayGoHackEnabled();
bool SkipNoticeScreen();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool RedZoneProtectionEnabled();
#endif

const Keymap& GetKeymap();
const std::vector<std::string>& GetGuestArguments();

} // namespace Config

#endif /* KYTY_COMMON_EMULATOR_CONFIG_H_ */
