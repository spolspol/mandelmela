// Direct computation kernel with graduated precision (PREC=1,2,3,4)
//
// Uses Metal function constants for compile-time specialisation:
// - PREC=1: float32 (~7 digits), zoom 0-4, fastest
// - PREC=2: double-float (~14 digits), zoom 4-10, fast
// - PREC=3: triple-float (~21 digits), zoom 10-16, medium
// - PREC=4: quad-float (~28 digits), zoom 16-22, slower
//
// Each precision level compiles to a separate optimised pipeline.
// Dead code elimination removes unused precision branches.
//
// Center coordinates packed via palette fields:
//   PREC=1: palette_base.xy = (center_re, center_im)
//   PREC=2: palette_base = (re_hi, re_lo, im_hi), palette_amp.x = im_lo
//   PREC=3: palette_base = re[0:3], palette_amp.xyz = (im[0], im[1], im[2])
//   PREC=4: palette_base.xyz = re[0:3], palette_amp = (re[3], im[0:2]), palette_phase.xy = im[2:4]
//
// Input: params buffer only (no reference orbit)
// Output: same format as perturbation kernel (frac_buf, de_data)

// Function constant for precision level (set at pipeline creation time)
constant int PREC [[function_constant(0)]];

kernel void mandelbrot_direct_prec_pass1(
    constant PerturbParams& params [[buffer(0)]],
    device float4* frac_buf [[buffer(2)]],         // Output: (fracIter, dist_est, stripe_t, 0)
    device float4* de_data [[buffer(3)]],          // Output: (dz_re, dz_im, z_re, z_im)
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(params.width) || gid.y >= uint(params.height)) return;

    int idx = gid.y * params.width + gid.x;

    // UV coordinates: normalise to [-0.5, 0.5] then apply aspect ratio
    float uv_x = (float(gid.x) + 0.5f) / float(params.width) - 0.5f;
    float uv_y = (float(gid.y) + 0.5f) / float(params.height) - 0.5f;
    uv_x *= params.ratio;

    // Apply rotation
    float sn = sin(params.angle);
    float cs = cos(params.angle);
    float ruv_x = cs * uv_x - sn * uv_y;
    float ruv_y = sn * uv_x + cs * uv_y;

    // Reconstruct scale
    int scale_exp = params.scale_exponent;

    // Variables for iteration (declared outside conditionals)
    float fracIter = -1.0f;  // -1 = inside set
    float dist_est_val = 0.0f;
    float stripe_t = 0.0f;
    float final_z_re = 0.0f, final_z_im = 0.0f;
    float dz_re = 0.0f, dz_im = 0.0f;

    int max_iter = params.iter;

    // ============================================================
    // PREC=1: Pure float32 (fastest, ~7 digits, zoom 0-4)
    // ============================================================
    if (PREC == 1) {
        // Unpack center from palette_base (repurposed)
        float center_re = params.palette_base.x;
        float center_im = params.palette_base.y;

        // Reconstruct scale as single float
        float scale = (scale_exp >= -126) ? ldexp(params.scale_hi, scale_exp) : 0.0f;

        // c = center + scale * ruv
        float cr = center_re + scale * ruv_x;
        float ci = center_im + scale * ruv_y;

        // z starts at 0
        float zr = 0.0f;
        float zi = 0.0f;

        // Stripe accumulator
        float stripe_sum = 0.0f, stripe_count = 0.0f, last_stripe = 0.0f;

        int i;
        for (i = 0; i < max_iter; i++) {
            float mag2 = zr * zr + zi * zi;
            if (mag2 > 256.0f) break;

            // DE derivative
            if (params.de_lighting == 1) {
                float new_dz_re = 2.0f * (zr * dz_re - zi * dz_im) + 1.0f;
                float new_dz_im = 2.0f * (zr * dz_im + zi * dz_re);
                dz_re = new_dz_re;
                dz_im = new_dz_im;
            }

            // Stripe accumulator
            if (params.stripe_intensity > 0.0f) {
                float stripe_val = 0.5f * sin(params.stripe_freq * atan2(zi, zr)) + 0.5f;
                last_stripe = stripe_val;
                stripe_sum += stripe_val;
                stripe_count += 1.0f;
            }

            // z = z^2 + c
            float zr_new = zr * zr - zi * zi + cr;
            zi = 2.0f * zr * zi + ci;
            zr = zr_new;
        }

        if (i < max_iter) {
            float dist = sqrt(zr * zr + zi * zi);
            // Protected smooth iteration (same as perturbation.metal)
            dist = max(dist, 2.72f);
            float log_dist = log(dist);
            log_dist = max(log_dist, 0.01f);
            fracIter = float(i) + 1.0f - log(log_dist) / log(2.0f);
            fracIter = clamp(fracIter, 0.0f, float(max_iter));

            if (params.de_lighting == 1) {
                float dz_mag = sqrt(dz_re * dz_re + dz_im * dz_im);
                dist_est_val = 2.0f * dist * log(dist) / max(dz_mag, 1e-10f);
            }

            if (params.stripe_intensity > 0.0f && stripe_count > 0.0f) {
                float stripe_avg = stripe_sum / stripe_count;
                float prev_avg = (stripe_count > 1.0f) ?
                    (stripe_sum - last_stripe) / (stripe_count - 1.0f) : stripe_avg;
                float frac = fract(fracIter);
                stripe_t = mix(prev_avg, stripe_avg, frac);
            }
        }

        final_z_re = zr;
        final_z_im = zi;
    }

    // ============================================================
    // PREC=2: Double-float (~14 digits, zoom 4-10)
    // ============================================================
    if (PREC == 2) {
        // Unpack center from palette fields (double-float)
        float2 center_re = float2(params.palette_base.x, params.palette_base.y);
        float2 center_im = float2(params.palette_base.z, params.palette_amp.x);

        // Reconstruct scale
        float scale_hi = ldexp(params.scale_hi, scale_exp);
        float scale_lo = (scale_exp >= -126) ? ldexp(params.scale_lo, scale_exp) : 0.0f;
        float2 scale = float2(scale_hi, scale_lo);

        // c = center + scale * ruv
        float2 offset_re = df_mul(scale, float2(ruv_x, 0.0f));
        float2 offset_im = df_mul(scale, float2(ruv_y, 0.0f));
        float2 cr = df_add(center_re, offset_re);
        float2 ci = df_add(center_im, offset_im);

        // z starts at 0
        float2 zr = float2(0.0f, 0.0f);
        float2 zi = float2(0.0f, 0.0f);

        // Stripe accumulator
        float stripe_sum = 0.0f, stripe_count = 0.0f, last_stripe = 0.0f;

        int i;
        for (i = 0; i < max_iter; i++) {
            float mag2 = zr.x * zr.x + zi.x * zi.x;
            if (mag2 > 256.0f) break;

            // DE derivative
            if (params.de_lighting == 1) {
                float new_dz_re = 2.0f * (zr.x * dz_re - zi.x * dz_im) + 1.0f;
                float new_dz_im = 2.0f * (zr.x * dz_im + zi.x * dz_re);
                dz_re = new_dz_re;
                dz_im = new_dz_im;
            }

            // Stripe accumulator
            if (params.stripe_intensity > 0.0f) {
                float stripe_val = 0.5f * sin(params.stripe_freq * atan2(zi.x, zr.x)) + 0.5f;
                last_stripe = stripe_val;
                stripe_sum += stripe_val;
                stripe_count += 1.0f;
            }

            // z = z^2 + c using double-float
            float2 zr2 = df_mul(zr, zr);
            float2 zi2 = df_mul(zi, zi);
            float2 zri = df_mul(zr, zi);

            zr = df_add(df_add(zr2, float2(-zi2.x, -zi2.y)), cr);
            zi = df_add(df_add(zri, zri), ci);
        }

        if (i < max_iter) {
            float dist = sqrt(zr.x * zr.x + zi.x * zi.x);
            // Protected smooth iteration (same as perturbation.metal)
            dist = max(dist, 2.72f);
            float log_dist = log(dist);
            log_dist = max(log_dist, 0.01f);
            fracIter = float(i) + 1.0f - log(log_dist) / log(2.0f);
            fracIter = clamp(fracIter, 0.0f, float(max_iter));

            if (params.de_lighting == 1) {
                float dz_mag = sqrt(dz_re * dz_re + dz_im * dz_im);
                dist_est_val = 2.0f * dist * log(dist) / max(dz_mag, 1e-10f);
            }

            if (params.stripe_intensity > 0.0f && stripe_count > 0.0f) {
                float stripe_avg = stripe_sum / stripe_count;
                float prev_avg = (stripe_count > 1.0f) ?
                    (stripe_sum - last_stripe) / (stripe_count - 1.0f) : stripe_avg;
                float frac = fract(fracIter);
                stripe_t = mix(prev_avg, stripe_avg, frac);
            }
        }

        final_z_re = zr.x;
        final_z_im = zi.x;
    }

    // ============================================================
    // PREC=3: Triple-float (~21 digits, zoom 10-16)
    // ============================================================
    if (PREC == 3) {
        // Unpack center from palette fields (triple-float)
        // palette_base.xyz = center_re[0:3]
        // palette_amp.xyz = center_im[0:3]
        float c_re[3], c_im[3];
        c_re[0] = params.palette_base.x;
        c_re[1] = params.palette_base.y;
        c_re[2] = params.palette_base.z;
        c_im[0] = params.palette_amp.x;
        c_im[1] = params.palette_amp.y;
        c_im[2] = params.palette_amp.z;

        // Compute offset (matches backup implementation)
        float offset_re[3] = {0.0f, 0.0f, 0.0f};
        float offset_im[3] = {0.0f, 0.0f, 0.0f};

        if (scale_exp >= -126) {
            float scale_value = ldexp(params.scale_hi, scale_exp);
            offset_re[0] = scale_value * ruv_x;
            offset_im[0] = scale_value * ruv_y;
        } else {
            // Deep zoom - shift to higher component
            int jumps = (-scale_exp - 126 + 22) / 23;
            int adjusted_exp = scale_exp + jumps * 23;

            if (jumps < 3) {
                float scale_value = ldexp(params.scale_hi, adjusted_exp);
                offset_re[jumps] = scale_value * ruv_x;
                offset_im[jumps] = scale_value * ruv_y;
            }
        }

        // c = center + offset
        tf_add(c_re, c_re, offset_re);
        tf_add(c_im, c_im, offset_im);

        // z starts at 0
        float z_re[3] = {0.0f, 0.0f, 0.0f};
        float z_im[3] = {0.0f, 0.0f, 0.0f};

        // Stripe accumulator
        float stripe_sum = 0.0f, stripe_count = 0.0f, last_stripe = 0.0f;

        // Temporaries
        float tmp1[3], tmp2[3], tmp3[3], neg_tmp[3];

        int i;
        for (i = 0; i < max_iter; i++) {
            float mag2 = z_re[0] * z_re[0] + z_im[0] * z_im[0];
            if (mag2 > 256.0f) break;

            // DE derivative
            if (params.de_lighting == 1) {
                float new_dz_re = 2.0f * (z_re[0] * dz_re - z_im[0] * dz_im) + 1.0f;
                float new_dz_im = 2.0f * (z_re[0] * dz_im + z_im[0] * dz_re);
                dz_re = new_dz_re;
                dz_im = new_dz_im;
            }

            // Stripe accumulator
            if (params.stripe_intensity > 0.0f) {
                float stripe_val = 0.5f * sin(params.stripe_freq * atan2(z_im[0], z_re[0])) + 0.5f;
                last_stripe = stripe_val;
                stripe_sum += stripe_val;
                stripe_count += 1.0f;
            }

            // z = z^2 + c using triple-float
            tf_mul(tmp1, z_re, z_re);         // tmp1 = z_re^2
            tf_mul(tmp2, z_im, z_im);         // tmp2 = z_im^2
            tf_mul(tmp3, z_re, z_im);         // tmp3 = z_re * z_im

            tf_neg(neg_tmp, tmp2);            // neg_tmp = -z_im^2
            tf_add(tmp2, tmp1, neg_tmp);      // tmp2 = z_re^2 - z_im^2
            tf_add(z_re, tmp2, c_re);         // z_re = (z_re^2 - z_im^2) + c_re

            tf_add(tmp1, tmp3, tmp3);         // tmp1 = 2 * z_re * z_im
            tf_add(z_im, tmp1, c_im);         // z_im = 2*z_re*z_im + c_im
        }

        if (i < max_iter) {
            float dist = sqrt(z_re[0] * z_re[0] + z_im[0] * z_im[0]);
            // Protected smooth iteration (same as perturbation.metal)
            dist = max(dist, 2.72f);
            float log_dist = log(dist);
            log_dist = max(log_dist, 0.01f);
            fracIter = float(i) + 1.0f - log(log_dist) / log(2.0f);
            fracIter = clamp(fracIter, 0.0f, float(max_iter));

            if (params.de_lighting == 1) {
                float dz_mag = sqrt(dz_re * dz_re + dz_im * dz_im);
                dist_est_val = 2.0f * dist * log(dist) / max(dz_mag, 1e-10f);
            }

            if (params.stripe_intensity > 0.0f && stripe_count > 0.0f) {
                float stripe_avg = stripe_sum / stripe_count;
                float prev_avg = (stripe_count > 1.0f) ?
                    (stripe_sum - last_stripe) / (stripe_count - 1.0f) : stripe_avg;
                float frac = fract(fracIter);
                stripe_t = mix(prev_avg, stripe_avg, frac);
            }
        }

        final_z_re = z_re[0];
        final_z_im = z_im[0];
    }

    // ============================================================
    // PREC=4: Quad-float (~28 digits, zoom 16-22)
    // ============================================================
    if (PREC == 4) {
        // Unpack center from palette fields (quad-float)
        // palette_base.xyz = center_re[0:3]
        // palette_amp.x    = center_re[3]
        // palette_amp.yz   = center_im[0:2]
        // palette_phase.xy = center_im[2:4]
        float c_re[4], c_im[4];
        c_re[0] = params.palette_base.x;
        c_re[1] = params.palette_base.y;
        c_re[2] = params.palette_base.z;
        c_re[3] = params.palette_amp.x;
        c_im[0] = params.palette_amp.y;
        c_im[1] = params.palette_amp.z;
        c_im[2] = params.palette_phase.x;
        c_im[3] = params.palette_phase.y;

        // Compute offset (matches backup implementation)
        float offset_re[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float offset_im[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        if (scale_exp >= -126) {
            float scale_value = ldexp(params.scale_hi, scale_exp);
            offset_re[0] = scale_value * ruv_x;
            offset_im[0] = scale_value * ruv_y;
        } else {
            // Deep zoom - shift to higher component
            int jumps = (-scale_exp - 126 + 22) / 23;
            int adjusted_exp = scale_exp + jumps * 23;

            if (jumps < 4) {
                float scale_value = ldexp(params.scale_hi, adjusted_exp);
                offset_re[jumps] = scale_value * ruv_x;
                offset_im[jumps] = scale_value * ruv_y;
            }
        }

        // c = center + offset
        qf_add(c_re, c_re, offset_re);
        qf_add(c_im, c_im, offset_im);

        // z starts at 0
        float z_re[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float z_im[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // Stripe accumulator
        float stripe_sum = 0.0f, stripe_count = 0.0f, last_stripe = 0.0f;

        // Temporaries
        float tmp1[4], tmp2[4], tmp3[4], neg_tmp[4];

        int i;
        for (i = 0; i < max_iter; i++) {
            float mag2 = z_re[0] * z_re[0] + z_im[0] * z_im[0];
            if (mag2 > 256.0f) break;

            // DE derivative
            if (params.de_lighting == 1) {
                float new_dz_re = 2.0f * (z_re[0] * dz_re - z_im[0] * dz_im) + 1.0f;
                float new_dz_im = 2.0f * (z_re[0] * dz_im + z_im[0] * dz_re);
                dz_re = new_dz_re;
                dz_im = new_dz_im;
            }

            // Stripe accumulator
            if (params.stripe_intensity > 0.0f) {
                float stripe_val = 0.5f * sin(params.stripe_freq * atan2(z_im[0], z_re[0])) + 0.5f;
                last_stripe = stripe_val;
                stripe_sum += stripe_val;
                stripe_count += 1.0f;
            }

            // z = z^2 + c using quad-float
            qf_mul(tmp1, z_re, z_re);         // tmp1 = z_re^2
            qf_mul(tmp2, z_im, z_im);         // tmp2 = z_im^2
            qf_mul(tmp3, z_re, z_im);         // tmp3 = z_re * z_im

            qf_neg(neg_tmp, tmp2);            // neg_tmp = -z_im^2
            qf_add(tmp2, tmp1, neg_tmp);      // tmp2 = z_re^2 - z_im^2
            qf_add(z_re, tmp2, c_re);         // z_re = (z_re^2 - z_im^2) + c_re

            qf_add(tmp1, tmp3, tmp3);         // tmp1 = 2 * z_re * z_im
            qf_add(z_im, tmp1, c_im);         // z_im = 2*z_re*z_im + c_im
        }

        if (i < max_iter) {
            float dist = sqrt(z_re[0] * z_re[0] + z_im[0] * z_im[0]);
            // Protected smooth iteration (same as perturbation.metal)
            dist = max(dist, 2.72f);
            float log_dist = log(dist);
            log_dist = max(log_dist, 0.01f);
            fracIter = float(i) + 1.0f - log(log_dist) / log(2.0f);
            fracIter = clamp(fracIter, 0.0f, float(max_iter));

            if (params.de_lighting == 1) {
                float dz_mag = sqrt(dz_re * dz_re + dz_im * dz_im);
                dist_est_val = 2.0f * dist * log(dist) / max(dz_mag, 1e-10f);
            }

            if (params.stripe_intensity > 0.0f && stripe_count > 0.0f) {
                float stripe_avg = stripe_sum / stripe_count;
                float prev_avg = (stripe_count > 1.0f) ?
                    (stripe_sum - last_stripe) / (stripe_count - 1.0f) : stripe_avg;
                float frac = fract(fracIter);
                stripe_t = mix(prev_avg, stripe_avg, frac);
            }
        }

        final_z_re = z_re[0];
        final_z_im = z_im[0];
    }

    frac_buf[idx] = float4(fracIter, dist_est_val, stripe_t, 0.0f);
    de_data[idx] = float4(dz_re, dz_im, final_z_re, final_z_im);
}
