// Pass 2: Visual effects and colouring
//
// Takes the fractal data from pass 1 and applies:
// - Colour mapping (RGB cosine palette or LCH space)
// - Distance estimation lighting
// - Ambient occlusion
// - Emboss effect
// - Metallic reflections
// - Detail boost

kernel void mandelbrot_perturb_pass2(
    constant PerturbParams& params [[buffer(0)]],
    device float4* frac_buf [[buffer(1)]],
    device float4* de_data [[buffer(2)]],
    device uchar4* output [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(params.width) || gid.y >= uint(params.height)) return;

    int idx = gid.y * params.width + gid.x;
    float fracIter = frac_buf[idx].x;

    // Inside the set
    if (fracIter < 0.0f) {
        output[idx] = uchar4(0, 0, 0, 255);
        return;
    }

    float dist_est_val = frac_buf[idx].y;
    float stripe_t = frac_buf[idx].z;

    // Screen-space derivatives via symmetric central differences
    // Use one-sided derivatives when neighbours are interior (negative fracIter)
    float dx = 0.0f, dy = 0.0f;
    if (gid.x > 0 && gid.x + 1 < uint(params.width)) {
        float left = frac_buf[idx - 1].x;
        float right = frac_buf[idx + 1].x;
        if (left >= 0.0f && right >= 0.0f) {
            dx = (right - left) * 0.5f;
        } else if (right >= 0.0f) {
            dx = right - fracIter;  // One-sided derivative
        } else if (left >= 0.0f) {
            dx = fracIter - left;   // One-sided derivative
        }
    } else if (gid.x + 1 < uint(params.width)) {
        float right = frac_buf[idx + 1].x;
        dx = (right >= 0.0f) ? right - fracIter : 0.0f;
    }
    if (gid.y > 0 && gid.y + 1 < uint(params.height)) {
        float above = frac_buf[idx - params.width].x;
        float below = frac_buf[idx + params.width].x;
        if (above >= 0.0f && below >= 0.0f) {
            dy = (below - above) * 0.5f;
        } else if (below >= 0.0f) {
            dy = below - fracIter;  // One-sided derivative
        } else if (above >= 0.0f) {
            dy = fracIter - above;  // One-sided derivative
        }
    } else if (gid.y + 1 < uint(params.height)) {
        float below = frac_buf[idx + params.width].x;
        dy = (below >= 0.0f) ? below - fracIter : 0.0f;
    }
    float gradMag = sqrt(dx * dx + dy * dy);

    // DE lighting derivatives
    float de_dx = 0.0f, de_dy = 0.0f;
    if (params.de_lighting == 1) {
        if (gid.x + 1 < uint(params.width)) {
            float right_de = frac_buf[idx + 1].y;
            de_dx = right_de - dist_est_val;
        }
        if (gid.y + 1 < uint(params.height)) {
            float below_de = frac_buf[idx + params.width].y;
            de_dy = below_de - dist_est_val;
        }
    }

    // Colouring base
    float freq = params.colour_freq / (1.0f + params.zoom_level * 0.5f);
    float t = fracIter * freq + params.time * 0.05f;

    // DE lighting setup
    float3 normal = float3(0.0f, 0.0f, 1.0f);
    float3 lightDir = normalize(float3(params.light_angle_x, params.light_angle_y, 1.0f));
    float de_light = 1.0f;
    float3 metallic_col = float3(1.0f);

    if (params.de_lighting == 1) {
        normal = normalize(float3(-de_dx, -de_dy, 1.0f));
        float diffuse = max(dot(normal, lightDir), 0.0f);
        float3 refl = reflect(-lightDir, normal);
        float specular = pow(max(refl.z, 0.0f), 16.0f);
        de_light = params.de_ambient + params.de_diffuse * diffuse + params.de_specular * specular;

        // Metallic reflections
        if (params.metallic > 0.0f) {
            float3 viewDir = float3(0.0f, 0.0f, 1.0f);
            float NdotV = max(dot(normal, viewDir), 0.0f);
            float fresnel = 0.7f + 0.3f * pow(1.0f - NdotV, 5.0f);
            float spec_metal = pow(max(dot(reflect(-lightDir, normal), viewDir), 0.0f), 96.0f);
            float envBright = 0.5f + 0.5f * normal.y;
            float3 envColour = mix(float3(0.15f, 0.15f, 0.2f), float3(0.9f, 0.95f, 1.0f), envBright);
            metallic_col = mix(float3(1.0f), envColour * fresnel + spec_metal * float3(1.0f), params.metallic);
        }
    }

    // Ambient occlusion
    float ao = 1.0f;
    if (params.ao_strength > 0.0f) {
        ao = 1.0f / (1.0f + params.ao_strength * gradMag * gradMag * 0.1f);
    }

    // Pre-compute emboss bump value (used by both colour paths)
    // Computed once to avoid duplicating trig and normalize operations
    float emboss_bump = 0.0f;
    if (params.emboss_strength > 0.0f) {
        float rad = params.emboss_angle * 0.017453293f;
        float2 light_dir = float2(cos(rad), sin(rad));
        float2 grad_n = normalize(float2(dx, dy) + float2(0.001f));
        emboss_bump = clamp(dot(grad_n, light_dir) * 2.0f, -1.0f, 1.0f);
    }

    float3 col;

    if (params.colour_mode == 1) {
        // LCH colour space
        float H = fmod(t * 360.0f, 360.0f);
        if (H < 0.0f) H += 360.0f;
        float L = params.lch_lightness;
        float C = params.lch_chroma;

        // Stripe modulation
        if (params.stripe_intensity > 0.0f) {
            L += (stripe_t - 0.5f) * params.stripe_intensity * 40.0f;
        }

        // Apply DE lighting
        L *= de_light;

        // Metallic effect
        if (params.metallic > 0.0f) {
            float metalLum = dot(metallic_col, float3(0.299f, 0.587f, 0.114f));
            L *= metalLum;
            C *= mix(1.0f, 0.5f, params.metallic);
        }

        // Ambient occlusion
        L *= ao;

        // Detail boost
        if (params.detail_boost > 0.0f) {
            L += params.detail_boost * min(gradMag * 10.0f, 30.0f);
        }

        // Emboss (using pre-computed bump)
        if (params.emboss_strength > 0.0f) {
            L += emboss_bump * 20.0f * params.emboss_strength;
        }

        L = clamp(L, 0.0f, 100.0f);
        col = lch_to_srgb(L, C, H);
    } else {
        // RGB cosine palette
        float cycle_freq = float(params.colour_count) / 3.0f;
        col = cosine_palette(t,
            params.palette_base,
            params.palette_amp,
            float3(1.0f) * cycle_freq,
            params.palette_phase
        );

        // Tonemap slightly
        float lum = dot(col, float3(0.299f, 0.587f, 0.114f));
        col = mix(col, float3(lum), 0.2f);
        col = pow(col, float3(0.95f));

        // Stripe modulation
        if (params.stripe_intensity > 0.0f) {
            col *= mix(1.0f, stripe_t * 2.0f, params.stripe_intensity);
        }

        // Apply DE lighting
        col *= de_light;

        // Metallic
        if (params.metallic > 0.0f) {
            col *= metallic_col;
        }

        // Ambient occlusion
        col *= ao;

        // Detail boost
        if (params.detail_boost > 0.0f) {
            float detailFactor = 1.0f + params.detail_boost * min(gradMag * 0.5f, 1.5f);
            col *= detailFactor;
        }

        col = clamp(col, 0.0f, 1.0f);

        // Emboss (using pre-computed bump)
        if (params.emboss_strength > 0.0f) {
            col += emboss_bump * 0.5f * params.emboss_strength;
            col = clamp(col, 0.0f, 1.0f);
        }
    }

    output[idx] = uchar4(
        uchar(col.x * 255.0f),
        uchar(col.y * 255.0f),
        uchar(col.z * 255.0f),
        255
    );
}
