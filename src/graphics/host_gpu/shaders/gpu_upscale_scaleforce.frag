// SPDX-FileCopyrightText: 2020 BreadFish64
// SPDX-FileCopyrightText: 2021 yuzu Emulator Project
// SPDX-License-Identifier: MIT
//
// ScaleForce upscaling filter, adapted from
// https://github.com/BreadFish64/ScaleFish/tree/master/scaleforce via the yuzu
// present shader, for the KytyPS5 present path.

#version 460 core

#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 frag_color;

layout(binding = 0) uniform sampler2D source;

const bool ignore_alpha = true;

float ColorDist1(vec4 a, vec4 b) {
	// https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.2020_conversion
	const vec3  K      = vec3(0.2627, 0.6780, 0.0593);
	const float scaleB = 0.5 / (1.0 - K.b);
	const float scaleR = 0.5 / (1.0 - K.r);
	vec4        diff   = a - b;
	float       Y      = dot(diff.rgb, K);
	float       Cb     = scaleB * (diff.b - Y);
	float       Cr     = scaleR * (diff.r - Y);
	vec3        YCbCr  = vec3(Y, Cb, Cr);
	float       d      = length(YCbCr);
	if (ignore_alpha) {
		return d;
	}
	return sqrt(a.a * b.a * d * d + diff.a * diff.a);
}

vec4 ColorDist(vec4 ref, vec4 A, vec4 B, vec4 C, vec4 D) {
	return vec4(ColorDist1(ref, A), ColorDist1(ref, B), ColorDist1(ref, C), ColorDist1(ref, D));
}

vec4 Scaleforce(sampler2D tex, vec2 tex_coord) {
	vec4 bl = textureOffset(tex, tex_coord, ivec2(-1, -1));
	vec4 bc = textureOffset(tex, tex_coord, ivec2(0, -1));
	vec4 br = textureOffset(tex, tex_coord, ivec2(1, -1));
	vec4 cl = textureOffset(tex, tex_coord, ivec2(-1, 0));
	vec4 cc = texture(tex, tex_coord);
	vec4 cr = textureOffset(tex, tex_coord, ivec2(1, 0));
	vec4 tl = textureOffset(tex, tex_coord, ivec2(-1, 1));
	vec4 tc = textureOffset(tex, tex_coord, ivec2(0, 1));
	vec4 tr = textureOffset(tex, tex_coord, ivec2(1, 1));

	vec4 offset_tl = ColorDist(cc, tl, tc, tr, cr);
	vec4 offset_br = ColorDist(cc, br, bc, bl, cl);

	// Calculate how different cc is from the texels around it
	const float plus_weight  = 1.5;
	const float cross_weight = 1.5;
	float       total_dist =
        dot(offset_tl + offset_br, vec4(cross_weight, plus_weight, cross_weight, plus_weight));

	if (total_dist == 0.0) {
		return cc;
	} else {
		// Add together all the distances with direction taken into account
		vec4  tmp          = offset_tl - offset_br;
		vec2  total_offset = tmp.wy * plus_weight + (tmp.zz + vec2(-tmp.x, tmp.x)) * cross_weight;

		// When the image has thin points, they tend to split apart.
		// This is because the texels all around are different and total_offset reaches into clear
		// areas. This works pretty well to keep the offset in bounds for these cases.
		float clamp_val    = length(total_offset) / total_dist;
		vec2  final_offset = vec2(clamp(total_offset, -clamp_val, clamp_val)) / textureSize(tex, 0);

		return texture(tex, tex_coord - final_offset);
	}
}

void main() {
	frag_color = Scaleforce(source, uv);
}
