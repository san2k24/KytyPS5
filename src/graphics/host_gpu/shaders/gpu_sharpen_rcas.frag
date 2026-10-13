// SPDX-FileCopyrightText: Copyright 2021 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// AMD FidelityFX Super Resolution 1.0 RCAS (Robust Contrast Adaptive
// Sharpening) stage for the KytyPS5 present path. Runs as a second pass after
// the upscaling filter (FSR EASU, ScaleForce or bicubic) when the user
// enables upscaling sharpening. Input is the already-upscaled intermediate
// target; output is the sharpened swapchain image.
//
// Const0.x holds the user sharpening strength (0-255). It is converted to the
// FSR1 "stops" scale here so the host does not have to reproduce FsrRcasCon's
// half-float packing. Const1 = (offset.x, offset.y, size.x, size.y) of the
// letterboxed present viewport inside the intermediate target.

#version 460 core
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require

layout(push_constant) uniform constants {
        uvec4 Const0;
        uvec4 Const1;
};

layout(set = 0, binding = 0) uniform sampler2D source;

#define A_GPU 1
#define A_GLSL 1

#include "ffx_a.h"

#define FSR_RCAS_F 1
#define FSR_RCAS_PASSTHROUGH_ALPHA 1
AF4 FsrRcasLoadF(ASU2 p) { return texelFetch(source, p, 0); }
void FsrRcasInputF(inout AF1 r, inout AF1 g, inout AF1 b) {}

#include "ffx_fsr1.h"

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

void main() {
        // 0-255 strength mapped onto FSR1's stops scale. FSR1 measures
        // sharpness in halvings ("stops"), where 0.0 = maximum sharpness, so
        // higher user strength must yield FEWER stops: 255 -> 0 stops (max),
        // 64 -> 3 stops (subtle), 0 -> 4 stops (very subtle).
        AF1 sharpness = (255.0 - AF1(Const0.x)) * (4.0 / 255.0);
        AU4 con;
        FsrRcasCon(con, sharpness);

        // Map the viewport-relative uv onto absolute intermediate texel
        // coordinates, clamped so edge texelFetches stay in bounds.
        ASU2 lo = ASU2(Const1.xy);
        ASU2 hi = lo + ASU2(Const1.zw) - ASU2(1, 1);
        ASU2 ip = clamp(ASU2(uv * vec2(Const1.zw)) + lo, lo, hi);
        AF3 c;
        AF1 a;
        FsrRcasF(c.r, c.g, c.b, a, AU2(ip), con);
        color = vec4(c, a);
}
