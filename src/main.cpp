#include "common/archive.h"
#include "common/common.h"
#include "common/dateTime.h"
#include "common/debug.h"
#include "common/file.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "emulator.h"
#include "kytyGitVersion.h"

#include <charconv>
#include <cstdio>
#include <filesystem>
#include <string_view>
#include <vector>
#include <fmt/format.h>
#include <magic_enum.hpp>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#define KYTY_WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#elif defined(__linux__)
#include <sched.h>
#endif

using namespace Common;
using namespace Emulator;

static std::string GetBuildString() {
        Date date = Date::FromMacros(std::string(__DATE__));

#if KYTY_BUILD == KYTY_BUILD_DEBUG
        std::string type = "Debug";
#elif KYTY_BUILD == KYTY_BUILD_RELEASE
        std::string type = "Release";
#else
        std::string type = "????";
#endif

        std::string compiler = Debug::GetCompiler() + "-" + Debug::GetLinker();

        std::string str =
            fmt::format("{}, {}, ver = {}, git = {}, date = {}", type.c_str(), compiler.c_str(),
                        KYTY_VERSION, KYTY_GIT_VERSION, date.ToString().c_str());

        return str;
}

static void PrintUsage() {
        ::printf("%s\n", GetBuildString().c_str());
        ::printf("kyty_emulator --game <dir|elf|zar> [options]\n\n");
        ::printf("Options:\n");
        ::printf("  --game <dir|elf|zar>                 Game directory, ELF, or ZArchive to load.\n");
        ::printf("  --game-patch <json>                  ETAHen cheat file.\n");
        ::printf("  --screen-width <num>                 Window width. Default: 1280.\n");
        ::printf("  --screen-height <num>                Window height. Default: 720.\n");
        ::printf(
            "  --user-name <name>                   Local user name (1-16 bytes). Default: Kyty.\n");
        ::printf("  --user-id <num>                      Local user ID. Default: %d.\n",
                 Config::DEFAULT_USER_ID);
        ::printf("  --mic <name>                        Capture from this microphone; omit for silence.\n");
        ::printf("  --controller-color <#RRGGBB>        Override the controller lightbar color.\n");
        ::printf("  --controller-volume <0-100>         DualSense speaker volume. Default: 50.\n");
        ::printf("  --controller-vibration <0-100>      DualSense vibration intensity. Default: 100.\n");
        ::printf(
            "  --present-mode <value>               Fifo, Mailbox, or Immediate. Default: Mailbox.\n");
        ::printf(
            "  --gpu <index>                        Vulkan physical device index. Default: auto.\n");
        ::printf("  --fullscreen                         Run in borderless desktop fullscreen.\n");
        ::printf("  --hdr                                Enable HDR10 output when supported.\n");
        ::printf(
            "  --hide-cursor                        Hide the cursor after 2 s idle. Default: off.\n");
        ::printf("  --no-splash                          Disable the startup splash screen.\n");
        ::printf("  --vr                                 Enable the virtual VR headset.\n");
        ::printf("  --amd-cpu                            Apply AMD CPU instruction patches.\n");
        ::printf("  --vblank-frequency <num>             Virtual vblank frequency. Default: 60.\n");
        ::printf("  --vblank-catchup <true|false>        Fire catch-up vblanks when host FPS drops. Default: true.\n");
        ::printf("  --speed-2x <true|false>              Run game at 2x speed (PPSSPP fast-forward). Default: false.\n");
        ::printf("  --frame-skip <1-10>                  Max vblanks fired per rendered frame. Default: 1.\n");
        ::printf("  --upscaling-filter <filter>         nearest, bilinear, bicubic, integer, fsr, or scaleforce. Default: bilinear.\n");
        ::printf("  --upscaling-sharpness <0-100>       FSR1 RCAS sharpening strength after the upscaling filter. 0 disables. Default: 80.\n");
        ::printf("  --texture-filtering <mode>          game (faithful) or force_af (force 16x anisotropic filtering). Default: game.\n");
        ::printf("  --aspect-ratio <ratio>              stretch, 16:9, 16:10, 4:3, or 21:9. Default: stretch.\n");
        ::printf("  --max-queued-frames <0-4>           Swapchain images; 0 = auto. 2 = lowest latency, 4 = smoothest.\n");
        ::printf("  --audio-output-device <name>        SDL playback device; omit for the system default.\n");
        ::printf("  --master-volume <0-200>             Main audio output volume in percent. Default: 100.\n");
        ::printf("  --shader-cache <true|false>         Persist the Vulkan pipeline cache per game. Default: true.\n");
        ::printf("  --shader-cache-async-save <bool>    Periodic background cache checkpoints so crashes keep warm runs. Default: true.\n");
        ::printf("  --thread-affinity <spec>            Host cores to run on, e.g. 0-7 or 0,2,4. Default: unrestricted.\n");
        ::printf("  --console-language <0-29>            Console language. Default: 1 (English US).\n");
        ::printf("  --vulkan-validation <true|false>     Enable Vulkan validation.\n");
        ::printf("  --gpu-assisted-validation <t|f>      Bounds-check shader accesses on the GPU.\n"
                 "                                       Implies --vulkan-validation; very slow.\n");
        ::printf("  --shader-validation <true|false>     Enable shader validation.\n");
        ::printf("  --tessellation                      Draw tessellation patches; skipped by default.\n");
        ::printf("  --shader-optimization-type <value>   None, Size, or Performance.\n");
        ::printf("  --shader-log-direction <value>       Silent, Console, or File.\n");
        ::printf("  --shader-log-folder <path>           Shader log output folder.\n");
        ::printf("  --command-buffer-dump <true|false>   Enable command buffer dumps.\n");
        ::printf("  --command-buffer-dump-folder <path>  Command buffer dump folder.\n");
        ::printf("  --graphics-debug-dump <true|false>   Enable graphics debug dumps.\n");
        ::printf("  --printf-direction <value>           Silent, Console, or File.\n");
        ::printf("  --printf-output-file <path>          Guest printf output file.\n");
        ::printf("  --profile                            Enable the Tracy profiler.\n");
        ::printf("  --spirv-debug-printf <true|false>    Enable SPIR-V debug printf.\n");
        ::printf(
            "  --readback-linear-images <true|false> Read back writable linear images on submit.\n");
        ::printf(
            "  --sync-raw-image-buffers <true|false> Synchronize raw reads of GPU images. Default: false.\n");
        ::printf(
            "  --trophy-notifications <true|false>   Show trophy unlock toasts and play their sound.\n");
        ::printf("  --playgo-hack                       Use the supplied PlayGo stub fallback.\n");
        ::printf(
            "  --skip-notice-screen <true|false>    Skip startup logos and notices in supported games.\n"
            "                                      Default: false.\n");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
        ::printf("  --redzone                            Protect the guest SysV red zone.\n");
#endif
        ::printf("  --keymap <Control=Input>             DualSense mapping; may be repeated.\n");
        ::printf("  --rd                                 Enable RenderDoc capture.\n");
}

static bool NextArg(int argc, char* argv[], int& index, std::string& out) {
        if (index + 1 >= argc) {
                return false;
        }

        index++;
        out = argv[index];
        return true;
}

static bool ParseBool(const std::string& value, bool& out) {
        if (Common::EqualNoCase(value, "true") || value == "1" || Common::EqualNoCase(value, "yes") ||
            Common::EqualNoCase(value, "on")) {
                out = true;
                return true;
        }

        if (Common::EqualNoCase(value, "false") || value == "0" || Common::EqualNoCase(value, "no") ||
            Common::EqualNoCase(value, "off")) {
                out = false;
                return true;
        }

        return false;
}

template <typename E>
static bool ParseEnum(const std::string& value, E& out) {
        auto enum_value = magic_enum::enum_cast<E>(value.c_str());
        if (!enum_value.has_value()) {
                return false;
        }

        out = enum_value.value();
        return true;
}

static bool ParseConsoleLanguage(const std::string& value, uint32_t& out) {
        uint32_t language = 0;
        auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), language);
        if (error != std::errc {} || end != value.data() + value.size() ||
            language > Config::MAX_CONSOLE_LANGUAGE) {
                return false;
        }
        out = language;
        return true;
}

static bool ParseUint32(const std::string& value, uint32_t& out) {
        uint32_t number   = 0;
        auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
        if (error != std::errc {} || end != value.data() + value.size()) {
                return false;
        }
        out = number;
        return true;
}

static bool ParseControllerColor(const std::string& value, Config::ControllerColor& out) {
        if (value.size() != 7 || value[0] != '#') {
                return false;
        }
        uint32_t rgb = 0;
        auto [end, error] = std::from_chars(value.data() + 1, value.data() + value.size(), rgb, 16);
        if (error != std::errc {} || end != value.data() + value.size()) {
                return false;
        }
        out = {static_cast<uint8_t>(rgb >> 16), static_cast<uint8_t>(rgb >> 8),
               static_cast<uint8_t>(rgb)};
        return true;
}

static bool ParseInt32(const std::string& value, int32_t& out) {
        int32_t number    = 0;
        auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
        if (error != std::errc {} || end != value.data() + value.size()) {
                return false;
        }
        out = number;
        return true;
}

static bool ParseUserId(const std::string& value, int32_t& out) {
        int32_t user_id   = 0;
        auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), user_id);
        if (error != std::errc {} || end != value.data() + value.size() ||
            !Config::IsConfiguredUserIdValid(user_id)) {
                return false;
        }
        out = user_id;
        return true;
}

// Accepts an empty specification (no restriction), a single core "3", a
// closed range "0-7", or a comma-separated mix of both: "0-3,8,9".
static bool ParseCpuAffinity(const std::string& spec, uint64_t& mask) {
        mask = 0;
        if (spec.empty()) {
                return true;
        }

        size_t pos = 0;
        while (pos <= spec.size()) {
                const size_t comma = spec.find(',', pos);
                const auto   token = spec.substr(pos, comma == std::string::npos ? std::string::npos
                                                                                 : comma - pos);
                if (token.empty()) {
                        return false;
                }

                const size_t dash = token.find('-');
                const auto   parse_core = [](const std::string& text, uint32_t& core) {
                        if (text.empty()) {
                                return false;
                        }
                        uint32_t value   = 0;
                        auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
                        return error == std::errc {} && end == text.data() + text.size() && value < 64
                                   ? (core = value, true)
                                   : false;
                };
                if (dash == std::string::npos) {
                        uint32_t core = 0;
                        if (!parse_core(token, core)) {
                                return false;
                        }
                        mask |= (uint64_t {1} << core);
                } else {
                        uint32_t lo = 0;
                        uint32_t hi = 0;
                        if (!parse_core(token.substr(0, dash), lo) ||
                            !parse_core(token.substr(dash + 1), hi) || lo > hi) {
                                return false;
                        }
                        for (uint32_t core = lo; core <= hi; core++) {
                                mask |= (uint64_t {1} << core);
                        }
                }

                if (comma == std::string::npos) {
                        break;
                }
                pos = comma + 1;
        }

        return mask != 0;
}

static bool ApplyCpuAffinity(const std::string& spec) {
        uint64_t mask = 0;
        if (!ParseCpuAffinity(spec, mask)) {
                ::printf("invalid thread-affinity: %s\n", spec.c_str());
                return false;
        }
        if (mask == 0) {
                return true;
        }

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
        if (!SetProcessAffinityMask(GetCurrentProcess(), static_cast<DWORD_PTR>(mask))) {
                ::printf("SetProcessAffinityMask failed (%lu)\n", GetLastError());
                return false;
        }
#elif defined(__linux__)
        cpu_set_t set;
        CPU_ZERO(&set);
        for (uint32_t core = 0; core < 64; core++) {
                if ((mask & (uint64_t {1} << core)) != 0) {
                        CPU_SET(core, &set);
                }
        }
        // Applied on the main thread before any worker thread is spawned;
        // Linux worker threads inherit the affinity of their creator.
        if (sched_setaffinity(0, sizeof(set), &set) != 0) {
                ::printf("sched_setaffinity failed\n");
                return false;
        }
#else
        ::printf("thread-affinity is not supported on this platform\n");
#endif
        return true;
}

static bool ParseArgs(int argc, char* argv[], RunOptions& options, bool& show_help) {
        show_help = false;

        for (int i = 1; i < argc; i++) {
                std::string arg = std::string(argv[i]);
                std::string value;

                if (arg == "--help" || arg == "-h") {
                        show_help = true;
                        continue;
                }

                if (arg == "--rd") {
                        options.config.renderdoc_enabled = true;
                        continue;
                }

                if (arg == "--fullscreen") {
                        options.config.fullscreen_enabled = true;
                        continue;
                }

                if (arg == "--hdr") {
                        options.config.hdr_enabled = true;
                        continue;
                }

                if (arg == "--hide-cursor") {
                        options.config.hide_cursor_enabled = true;
                        continue;
                }

                if (arg == "--no-splash") {
                        options.config.splash_enabled = false;
                        continue;
                }

                if (arg == "--vr") {
                        options.config.vr_enabled = true;
                        continue;
                }

                if (arg == "--amd-cpu") {
                        options.config.amd_cpu_enabled = true;
                        continue;
                }

                if (arg == "--playgo-hack") {
                        options.config.playgo_hack_enabled = true;
                        continue;
                }

                if (arg == "--tessellation") {
                        options.config.tessellation_enabled = true;
                        continue;
                }

                if (arg == "--profile") {
                        options.config.profiler_enabled = true;
                        continue;
                }

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
                if (arg == "--redzone") {
                        options.config.red_zone_protection_enabled = true;
                        continue;
                }
#endif

                if (!arg.starts_with("--")) {
                        ::printf("game input must be provided with --game\n");
                        return false;
                }

                if (!NextArg(argc, argv, i, value)) {
                        ::printf("missing value for %s\n", arg.c_str());
                        return false;
                }

                if (arg == "--game") {
                        if (!options.app0_dir.empty()) {
                                ::printf("--game can only be specified once\n");
                                return false;
                        }

                        value           = Common::FixFilenameSlash(value);
                        const auto path = Common::PathFromUtf8(value);

                        if (Common::File::IsDirectoryExisting(path)) {
                                options.app0_dir = path;
                                options.elf      = "/app0/eboot.bin";
                        } else if (Common::IsSupportedArchive(path) && Common::File::IsFileExisting(path)) {
                                const auto root = Common::MakeArchivePath(path);
                                if (!Common::File::IsFileExisting(root / "eboot.bin")) {
                                        ::printf("Archive does not contain eboot.bin: %s\n", value.c_str());
                                        return false;
                                }
                                options.app0_dir = root;
                                options.elf      = "/app0/eboot.bin";
                        } else if (Common::File::IsFileExisting(path)) {
                                options.app0_dir = path.parent_path();

                                if (options.app0_dir.empty()) {
                                        options.app0_dir = ".";
                                }

                                options.elf = std::filesystem::path("/app0") / path.filename();
                        } else {
                                ::printf("--game must point to an existing directory, ELF, or archive: %s\n",
                                         value.c_str());
                                return false;
                        }
                } else if (arg == "--guest-exec") {
                        options.elf = Common::PathFromUtf8(Common::FixFilenameSlash(value));
                        options.config.guest_arguments.assign(argv + i + 1, argv + argc);
                        break;
                } else if (arg == "--game-patch") {
                        if (!options.game_patch.empty()) {
                                ::printf("--game-patch can only be specified once\n");
                                return false;
                        }
                        value = Common::FixFilenameSlash(value);
                        const auto path = Common::PathFromUtf8(value);

                        if (!Common::File::IsFileExisting(path)) {
                                ::printf("--game-patch must point to an existing file: %s\n", value.c_str());
                                return false;
                        }
                        options.game_patch = path;
                } else if (arg == "--screen-width") {
                        if (!ParseUint32(value, options.config.screen_width) ||
                            options.config.screen_width == 0) {
                                ::printf("invalid screen width: %s\n", value.c_str());
                                return false;
                        }
                } else if (arg == "--screen-height") {
                        if (!ParseUint32(value, options.config.screen_height) ||
                            options.config.screen_height == 0) {
                                ::printf("invalid screen height: %s\n", value.c_str());
                                return false;
                        }
                } else if (arg == "--user-name") {
                        if (value.empty() || value.size() > Config::MAX_USER_NAME_LENGTH) {
                                ::printf("invalid user name: must contain 1-%zu bytes\n",
                                         Config::MAX_USER_NAME_LENGTH);
                                return false;
                        }
                        options.config.user_name = value;
                } else if (arg == "--user-id") {
                        if (!ParseUserId(value, options.config.user_id)) {
                                ::printf("invalid user ID: %s\n", value.c_str());
                                return false;
                        }
                } else if (arg == "--mic") {
                        options.config.audio_input_device = value;
                } else if (arg == "--controller-color") {
                        Config::ControllerColor color {};
                        if (!ParseControllerColor(value, color)) {
                                ::printf("invalid controller color (expected #RRGGBB): %s\n", value.c_str());
                                return false;
                        }
                        options.config.controller_color = color;
                } else if (arg == "--controller-volume") {
                        if (!ParseUint32(value, options.config.controller_speaker_volume) ||
                            options.config.controller_speaker_volume > 100) {
                                ::printf("invalid controller volume: %s\n", value.c_str());
                                return false;
                        }
                } else if (arg == "--controller-vibration") {
                        if (!ParseUint32(value, options.config.controller_vibration_intensity) ||
                            options.config.controller_vibration_intensity > 100) {
                                ::printf("invalid controller vibration intensity: %s\n", value.c_str());
                                return false;
                        }
                } else if (arg == "--present-mode") {
                        if (!ParseEnum(value, options.config.present_mode)) {
                                ::printf("invalid present mode: %s\n", value.c_str());
                                return false;
                        }
                } else if (arg == "--gpu") {
                        if (!ParseInt32(value, options.config.gpu_index)) {
                                ::printf("invalid gpu index: %s\n", value.c_str());
                                return false;
                        }
                } else if (arg == "--vblank-frequency") {
                        if (!ParseUint32(value, options.config.vblank_frequency)) {
                                ::printf("invalid vblank frequency: %s\n", value.c_str());
                                return false;
                        }
                } else if (arg == "--vblank-catchup") {
                        bool v = false;
                        if (!ParseBool(value, v)) {
                                ::printf("invalid vblank-catchup: %s\n", value.c_str());
                                return false;
                        }
                        options.config.vblank_catchup_enabled = v;
                } else if (arg == "--speed-2x") {
                        bool v = false;
                        if (!ParseBool(value, v)) {
                                ::printf("invalid speed-2x: %s\n", value.c_str());
                                return false;
                        }
                        options.config.speed_2x_enabled = v;
                } else if (arg == "--frame-skip") {
                        uint32_t n = 0;
                        if (!ParseUint32(value, n) || n < 1 || n > 10) {
                                ::printf("invalid frame-skip: %s\n", value.c_str());
                                return false;
                        }
                        options.config.frame_skip_count = n;
                } else if (arg == "--upscaling-filter") {
                        if (value == "nearest" || value == "bilinear" || value == "bicubic" ||
                            value == "integer" || value == "fsr" || value == "scaleforce") {
                                options.config.upscaling_filter = value;
                        } else {
                                ::printf("invalid upscaling-filter: %s\n", value.c_str());
                                return false;
                        }
                } else if (arg == "--upscaling-sharpness") {
                        if (!ParseUint32(value, options.config.upscaling_sharpness) ||
                            options.config.upscaling_sharpness > 100) {
                                ::printf("invalid upscaling-sharpness: %s\n", value.c_str());
                                return false;
                        }
                } else if (arg == "--texture-filtering") {
                        if (value == "game" || value == "force_af") {
                                options.config.texture_filtering = value;
                        } else {
                                ::printf("invalid texture-filtering: %s\n", value.c_str());
                                return false;
                        }
                } else if (arg == "--aspect-ratio") {
                        if (value == "stretch" || value == "16:9" || value == "16:10" ||
                            value == "4:3" || value == "21:9") {
                                options.config.aspect_ratio = value;
                        } else {
                                ::printf("invalid aspect-ratio: %s\n", value.c_str());
                                return false;
                        }
                } else if (arg == "--max-queued-frames") {
                        if (!ParseUint32(value, options.config.max_queued_frames) ||
                            options.config.max_queued_frames > 4) {
                                ::printf("invalid max-queued-frames: %s\n", value.c_str());
                                return false;
                        }
                } else if (arg == "--audio-output-device") {
                        options.config.audio_output_device = value;
                } else if (arg == "--master-volume") {
                        if (!ParseUint32(value, options.config.master_volume) ||
                            options.config.master_volume > 200) {
                                ::printf("invalid master-volume: %s\n", value.c_str());
                                return false;
                        }
                } else if (arg == "--shader-cache") {
                        if (!ParseBool(value, options.config.shader_cache_enabled)) {
                                ::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
                                return false;
                        }
                } else if (arg == "--shader-cache-async-save") {
                        if (!ParseBool(value, options.config.shader_cache_async_save)) {
                                ::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
                                return false;
                        }
                } else if (arg == "--thread-affinity") {
                        options.config.cpu_affinity = value;
                } else if (arg == "--console-language") {
                        if (!ParseConsoleLanguage(value, options.config.console_language)) {
                                ::printf("invalid console language: %s\n", value.c_str());
                                return false;
                        }
                } else if (arg == "--vulkan-validation") {
                        if (!ParseBool(value, options.config.vulkan_validation_enabled)) {
                                ::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
                                return false;
                        }
                } else if (arg == "--gpu-assisted-validation") {
                        if (!ParseBool(value, options.config.gpu_assisted_validation_enabled)) {
                                ::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
                                return false;
                        }
                } else if (arg == "--shader-validation") {
                        if (!ParseBool(value, options.config.shader_validation_enabled)) {
                                ::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
                                return false;
                        }
                } else if (arg == "--shader-optimization-type") {
                        if (!ParseEnum(value, options.config.shader_optimization_type)) {
                                ::printf("invalid shader optimization type: %s\n", value.c_str());
                                return false;
                        }
                } else if (arg == "--shader-log-direction") {
                        if (!ParseEnum(value, options.config.shader_log_direction)) {
                                ::printf("invalid shader log direction: %s\n", value.c_str());
                                return false;
                        }
                } else if (arg == "--shader-log-folder") {
                        options.config.shader_log_folder = Common::PathFromUtf8(value);
                } else if (arg == "--command-buffer-dump") {
                        if (!ParseBool(value, options.config.command_buffer_dump_enabled)) {
                                ::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
                                return false;
                        }
                } else if (arg == "--command-buffer-dump-folder") {
                        options.config.command_buffer_dump_folder = Common::PathFromUtf8(value);
                } else if (arg == "--graphics-debug-dump") {
                        if (!ParseBool(value, options.config.graphics_debug_dump_enabled)) {
                                ::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
                                return false;
                        }
                } else if (arg == "--printf-direction") {
                        if (!ParseEnum(value, options.config.printf_direction)) {
                                ::printf("invalid printf direction: %s\n", value.c_str());
                                return false;
                        }
                } else if (arg == "--printf-output-file") {
                        options.config.printf_output_file = Common::PathFromUtf8(value);
                } else if (arg == "--spirv-debug-printf") {
                        if (!ParseBool(value, options.config.spirv_debug_printf_enabled)) {
                                ::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
                                return false;
                        }
                } else if (arg == "--readback-linear-images") {
                        if (!ParseBool(value, options.config.readback_linear_images)) {
                                ::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
                                return false;
                        }
                } else if (arg == "--sync-raw-image-buffers") {
                        if (!ParseBool(value, options.config.sync_raw_image_buffers)) {
                                ::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
                                return false;
                        }
                } else if (arg == "--trophy-notifications") {
                        if (!ParseBool(value, options.config.trophy_enabled)) {
                                ::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
                                return false;
                        }
                } else if (arg == "--skip-notice-screen") {
                        if (!ParseBool(value, options.config.skip_notice_screen)) {
                                ::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
                                return false;
                        }
                } else if (arg == "--keymap") {
                        const auto split = value.find('=');
                        if (split == std::string::npos || split == 0 || split + 1 == value.size()) {
                                ::printf("invalid keymap: %s\n", value.c_str());
                                return false;
                        }
                        options.config.keymap.push_back(value);
                } else {
                        ::printf("unknown option: %s\n", arg.c_str());
                        return false;
                }
        }

        if (options.config.gpu_assisted_validation_enabled) {
                options.config.vulkan_validation_enabled = true;
        }

        return show_help || (!options.app0_dir.empty() && !options.elf.empty());
}

static int Main(int argc, char* argv[]) {
        VirtualMemory::Init();
        InitializeThreads();

        RunOptions options;
        bool       show_help = false;

        if (argc < 2) {
                PrintUsage();
                return 0;
        }

        if (!ParseArgs(argc, argv, options, show_help)) {
                PrintUsage();
                return 1;
        }

        if (show_help) {
                PrintUsage();
                return 0;
        }

        // Must happen before Run(): every emulation thread is created inside
        // Run(), and workers inherit the affinity of their creator on Linux.
        if (!ApplyCpuAffinity(options.config.cpu_affinity)) {
                return 1;
        }

        Run(options);

        return 0;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

int wmain(int argc, wchar_t* argv[]) {
    std::vector<std::string> utf8_args;
    utf8_args.reserve(static_cast<size_t>(argc));

    for (int index = 0; index < argc; index++) {
        const std::wstring_view wide(argv[index]);
        const std::u16string utf16(wide.begin(), wide.end());

        utf8_args.push_back(Common::Utf16ToUtf8(utf16));
    }

    std::vector<char*> utf8_argv;
    utf8_argv.reserve(utf8_args.size());

    for (auto& argument: utf8_args) {
        utf8_argv.push_back(argument.data());
    }

    return Main(argc, utf8_argv.data());
}

#else

int main(int argc, char* argv[]) {
    return Main(argc, argv);
}

#endif
