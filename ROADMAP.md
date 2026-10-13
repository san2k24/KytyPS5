# KytyPS5 Fork — Architecture Roadmap

Status-quo baseline of this fork (branch `main`) and the planned evolution of its
graphics, CPU and tooling layers. Items are ordered by impact/effort ratio for
playability on mid-range Windows hardware.

## Shipped in this fork

| Area | Feature | Notes |
|------|---------|-------|
| Present | FSR 1.0 EASU / ScaleForce / Bicubic / integer upscaling | swapchain shader path, letterbox + aspect ratio aware |
| Present | **FSR 1.0 RCAS sharpening** (new) | second pass over an intermediate target, 0–100 % strength, works after fsr/scaleforce/bicubic |
| Textures | **Forced 16x anisotropic filtering** (new) | `texture_filtering=force_af` upgrades game bilinear samplers; compare/unorm samplers untouched |
| Shaders | Persistent Vulkan driver pipeline cache | `_PipelineCache/<title>.bin`, signature + XXH3 validated |
| Shaders | **Crash-safe cache checkpoints** (new) | every 2 min, blob fetched on the GPU thread, file written on a background thread; crashes no longer lose the warm cache |
| CPU | AMD CPU compatibility (VRSQRTPS/EXTRQ/INSERTQ/RDPID/CLWB) | in-place traps + trampolines + red-zone liveness (upstream machinery) |
| CPU | Thread affinity, frame skip, vblank catch-up, 2x speed | |
| Audio | Master volume, output device selection (SDL3) | |
| Diagnostics | Guest fault context dump, module+RVA symbolization (new), stack-overflow hint (new), nested-fault guard in the x64 trap emulator (new) | the wild `pc=0x5ffffffff` class now reports *which module* jumped |

## Short term (next iterations)

1. **True asynchronous shader/pipeline compilation (yuzu-style).**
   The renderer currently compiles guest shader permutations synchronously on
   the GPU command thread. The plan mirrors yuzu:
   - worker pool translating GLC → SPIR-V off-thread (recompiler is already
     context-free per program; the SRT runtime reads must be snapshotted);
   - placeholder-pipeline draw skipping (record nothing for a draw whose
     pipeline is not ready, replay on the next frame);
   - a `shader_async_compilation` toggle (default on) plus a per-game override.
   Risk: command-recorder assumptions around compute dispatches and event
   writes; needs a dedicated sync-audit pass before landing.

2. **Persisted SPIR-V translation cache.**
   Key = hash(source program + specialization + options). Cuts warm-run
   translation entirely instead of only the driver-side compile. Requires a
   stable serialization of `CompileOptions` and the resource plan.

3. **RCAS dials.** Expose FSR1's denoise flag and per-game sharpness presets;
   add a hotkey to cycle strength at runtime.

4. **Resolution scaling.** Render-target scale factor (0.5x–2.0x) applied at
   the GNM render-target level rather than only at present, with the existing
   EASU/RCAS chain used for the final resolve.

## Mid term

5. **Frame generation (FSR 3 style).** Requires motion vectors, which the GNM
   command processor does not track today. Work items: per-draw MV export pass
   (optional in the recompiler), UI/HUD exclusion heuristics, async
   interpolation queue on a second Vulkan queue. High effort, high reward.

6. **HDR pipeline improvements.** The swapchain already supports HDR10 (PQ);
   missing: game gamma/tonemap override hooks and auto-detection of HDR
   content per title.

7. **FSR 4 / hardware upscalers.** FSR 4 is AMD's ML-based upscaler tied to
   RDNA 4 hardware and a closed SDK; it cannot be embedded portably in an
   emulator. The practical path is: keep FSR 1 (spatial, universal), add FSR 2
   (temporal) once motion vectors exist (item 5), and optionally expose
   vendor-specific paths (DLSS via NVNGX proxy) behind a plugin interface.

8. **OpenGL backend — not planned.** KytyPS5 is architected around Vulkan
   features that OpenGL cannot express: bindless BDA shader memory access,
   dynamic rendering, per-page GPU fault tracking and compute-based tiling
   emulation. An OpenGL backend would mean a second full renderer (GLSL
   translation of every recompiled shader, reworked descriptor model) with
   strictly worse compatibility. Effort is better spent on Vulkan correctness
   and the async pipeline above. This is a deliberate architectural decision,
   not an oversight.

## CPU / stability

9. **Red-zone safety audit.** The Windows exception dispatcher clobbers ~1.2 KB
   below RSP when delivering the traps used by the AMD patcher. The upstream
   liveness analysis covers decoded functions, but wild jumps (e.g.
   `pc=0x00000005ffffffff`) observed in the field are consistent with a missed
   live-range. Follow-ups:
   - instrument trap sites (per-site counters) to correlate traps with later
     corruption;
   - default the *per-game* AMD compatibility toggle to OFF for titles that
     crash, and document the workaround (game settings → uncheck
     "AMD CPU compatibility");
   - long term: relocate *every* trapped instruction into a red-zone-safe
     trampoline instead of trapping in place.

10. **Crash reporting.** With module+RVA symbolization now in the fault dump,
    add: automatic guest backtrace unwind (eh_frame based), PDB upload in CI,
    and a crash-bucket hash so repeated reports deduplicate.

## Build / release

11. Keep the Windows-only CI lane green on every push; release artifacts stay
    Windows x64 only (launcher + emulator + PDBs in a separate zip).
