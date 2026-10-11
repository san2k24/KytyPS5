# `VK_KHR_fragment_shader_barycentric` fallback

This fork ships a runtime fallback that lets **KytyPS5** boot on Vulkan devices
that do **not** advertise `VK_KHR_fragment_shader_barycentric`. The most common
case is AMD **RDNA1** GPUs such as the **Radeon RX 5500 XT**, which crash on
startup with the upstream build:

```
--- Error ---
Could not find suitable device:
Radeon RX 5500 XT: fragmentShaderBarycentric is not supported
in vulkanWindow.cpp:966
```

The fallback keeps the upstream code path untouched when the host device
supports the extension, and only kicks in for GPUs that lack it. The trade-off
is documented below.

## Why the upstream build rejects RDNA1

`VK_KHR_fragment_shader_barycentric` exposes the SPIR-V builtin
`BaryCoordKHR` so that the fragment shader can read the per-fragment
barycentric weights (the (u, v, w) of the fragment within its triangle). The
PS5 GPU does this in hardware, and KytyPS5's shader recompiler translates the
PS5 `ps_perspective_*` VGPRs directly to those SPIR-V builtins.

RDNA1 hardware (Navi 14 / Navi 10) does **not** implement this Vulkan
extension at the driver level, so the upstream `vulkanWindow.cpp` rejects the
device during selection.

## What this fork changes

### 1. Device extension is optional

`vulkanWindow.cpp` no longer adds
`VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME` to the list of *required*
device extensions. Instead, after a physical device is selected, the
presenter enumerates the device's available extensions and only enables the
extension when it is actually advertised.

### 2. Feature check is a log, not a reject

The `check_feature(fragment_barycentric.fragmentShaderBarycentric, ...)`
call that used to abort device selection now only logs a warning. Devices
without the extension are still accepted.

### 3. Feature enable is conditional

`VulkanCreateDevice` only requests
`fragment_barycentric.fragmentShaderBarycentric = VK_TRUE` when the chosen
device actually supports the extension. The `pNext` chain is built
accordingly so the driver never sees a feature struct for an extension it
does not advertise.

### 4. Runtime flag propagates to the SPIR-V backend

`GraphicContext::fragment_barycentric_enabled` is a new bool that records the
host support state. Right after `vkCreateDevice` succeeds, the presenter calls
`ShaderRecompiler::SetHostSupportsFragmentBarycentric(bool)` so the SPIR-V
emitter can read it at shader compile time via
`ShaderRecompiler::HostSupportsFragmentBarycentric()`.

This is a process-wide flag set **once** per session. It must remain stable
because the SPIR-V emitted for a shader is cached for the process lifetime.

### 5. SPIR-V emitter skips the capability when unsupported

`spirvEmitterModule.cpp` only emits:

- `spv::CapabilityFragmentBarycentricKHR`
- `SPV_KHR_fragment_shader_barycentric` extension

when `HostSupportsFragmentBarycentric()` returns true. Otherwise the
capability and extension are omitted, so `vkCreateShaderModule` accepts the
SPIR-V on RDNA1.

### 6. Per-vertex interpolation falls back to flat-shading

When a pixel shader declares per-vertex attribute interpolation (the case
that needs barycentric weights), `spirvEmitterFlow.cpp::EmitAttribute` checks
the host support flag:

- **Host supports barycentric**: existing code path, weights come from the
  `BaryCoordKHR` builtin.
- **Host does NOT support barycentric**: emits flat-shading by taking the
  first vertex's value. No `OpAccessChain` against the undeclared barycentric
  variable, no SPIR-V validator error.

## Behavior matrix

| Host GPU                                | Extension advertised? | Code path                                       | Visual accuracy       |
| --------------------------------------- | --------------------- | ----------------------------------------------- | --------------------- |
| AMD RDNA2+ (RX 6000+), NVIDIA Turing+, Intel Arc+ | Yes         | Upstream path with `BaryCoordKHR` builtin      | Full                  |
| AMD RDNA1 (RX 5000 series), older GPUs | No                    | Fallback path with flat-shading for per-vertex  | Minor glitches in games using barycentric interpolation |

## Practical impact on RX 5500 XT

- **Boot**: the emulator reaches the splash/menu screen instead of aborting
  with `Could not find suitable device`.
- **Games using simple pixel shaders** (no per-vertex interpolation, no
  barycentric-dependent techniques): render correctly.
- **Games using per-vertex attribute interpolation**: visuals may show
  faceted/wrong colors where smooth interpolation was expected. Geometry is
  still correct; only the colors look off.
- **Performance**: no measurable overhead — the fallback emits fewer
  instructions than the original weighted-sum.

## Recommended steps before testing

1. **Update your AMD driver** to the latest Adrenalin release from
   https://www.amd.com/en/support — newer driver versions occasionally
   advertise previously-missing Vulkan extensions even on RDNA1 hardware.
2. **Clear the shader cache** (`_Shaders/` in the working directory) on the
   first run after switching GPU, since cached SPIR-V may contain the
   capability decoration and fail to load on the new device.
3. **Disable Vulkan validation layers** in `Config` if you had them enabled
   for debugging — validation is stricter about undeclared capabilities.
