// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Single-pass AMD FidelityFX Super Resolution 1.0 (EASU stage) adapted for the
// KytyPS5 present path. The EASU (edge-adaptive spatial upsampling) pass runs
// while blitting the guest frame into the swapchain; the optional RCAS
// sharpening stage of FSR 1.0 would need an intermediate target and is not
// part of this pass.
//
// AMD FidelityFX SUPER RESOLUTION
// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. (MIT)

#version 460 core
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Const0..Const3 hold the FsrEasuCon() constants; Const4.xy is the size of the
// destination (letterboxed) viewport in pixels.
layout(push_constant) uniform constants {
	uvec4 Const0;
	uvec4 Const1;
	uvec4 Const2;
	uvec4 Const3;
	uvec4 Const4;
};

layout(set = 0, binding = 0) uniform sampler2D source;

#define A_GPU 1
#define A_GLSL 1

#include "ffx_a.h"

#define FSR_EASU_F 1
AF4 FsrEasuRF(AF2 p) { return textureGather(source, p, 0); }
AF4 FsrEasuGF(AF2 p) { return textureGather(source, p, 1); }
AF4 FsrEasuBF(AF2 p) { return textureGather(source, p, 2); }

#include "ffx_fsr1.h"

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

void main() {
	// uv spans the letterboxed destination viewport, so uv * dst_size yields
	// the output pixel position (at pixel centers) EASU expects.
	AF3 c;
	FsrEasuF(c, AU2(uv * vec2(Const4.xy)), Const0, Const1, Const2, Const3);
	color = vec4(c, texture(source, uv).a);
}
