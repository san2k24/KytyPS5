// SPDX-FileCopyrightText: Copyright 2025 KytyPS5 contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Catmull-Rom bicubic upscaling filter for the KytyPS5 present path.
// Classic 16-tap separable Catmull-Rom: sharp on edges, no ringing beyond the
// standard -0.5 kernel overshoot. Uses textureSize() so no constants are
// required from the pipeline.

#version 460 core

#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 frag_color;

layout(binding = 0) uniform sampler2D source;

vec4 catmull_rom_weights(float x) {
        float x2 = x * x;
        float x3 = x2 * x;
        return vec4(-0.5 * x3 + x2 - 0.5 * x, 1.5 * x3 - 2.5 * x2 + 1.0,
                    -1.5 * x3 + 2.0 * x2 + 0.5 * x, 0.5 * x3 - 0.5 * x2);
}

vec4 sample_bicubic(sampler2D tex, vec2 tex_coord) {
        const vec2 tex_size = vec2(textureSize(tex, 0));
        // Position in texel space, shifted so that texel centers sit at integers.
        const vec2 coord = tex_coord * tex_size - 0.5;
        const vec2 fex   = fract(coord);
        // Texel center of the (0, 0) tap of the 4x4 Catmull-Rom neighborhood.
        const vec2 base = (floor(coord) + 0.5) / tex_size;
        const vec2 step = 1.0 / tex_size;

        const vec4 wx = catmull_rom_weights(fex.x);
        const vec4 wy = catmull_rom_weights(fex.y);

        vec4 result = vec4(0.0);
        for (int row = 0; row < 4; row++) {
                const float y_off = (float(row) - 1.0) * step.y;
                const vec4  taps  = vec4(texture(tex, base + vec2(-step.x, y_off)),
                                 texture(tex, base + vec2(0.0, y_off)),
                                 texture(tex, base + vec2(step.x, y_off)),
                                 texture(tex, base + vec2(2.0 * step.x, y_off)));
                // Weighted row sum: dot of horizontal weights with the four taps.
                result += taps * wx * wy[row];
        }
        return result;
}

void main() {
        frag_color = sample_bicubic(source, uv);
}
