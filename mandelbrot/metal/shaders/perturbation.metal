// Perturbation theory Mandelbrot kernel (two-pass)
//
// Uses SCALED double-float arithmetic for deltas
// Enables zoom depths beyond float32's 1e±38 range limit (1e1000+ possible)
//
// Outputs: frac_buf (fracIter, dist_est, stripe_t, 0), de_data (dz_re, dz_im, z_re, z_im)

kernel void mandelbrot_perturb_pass1(
    constant PerturbParams& params [[buffer(0)]],
    device const float4* ref_orbit [[buffer(1)]],  // Reference orbit: (re_hi, re_lo, im_hi, im_lo)
    device float4* frac_buf [[buffer(2)]],         // Output: (fracIter, dist_est, stripe_t, 0)
    device float4* de_data [[buffer(3)]],          // Output: (dz_re, dz_im, z_re, z_im)
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(params.width) || gid.y >= uint(params.height)) return;

    int idx = gid.y * params.width + gid.x;

    // UV coordinates: normalise to [-0.5, 0.5] then apply aspect ratio
    float uv_x = (float(gid.x) + 0.5) / float(params.width) - 0.5;
    float uv_y = (float(gid.y) + 0.5) / float(params.height) - 0.5;
    uv_x *= params.ratio;

    // Apply rotation
    float sn = sin(params.angle);
    float cs = cos(params.angle);
    float ruv_x = cs * uv_x - sn * uv_y;
    float ruv_y = sn * uv_x + cs * uv_y;

    // Delta c as SCALED double-float
    ScaledDF scale_sdf;
    scale_sdf.mantissa = float2(params.scale_hi, params.scale_lo);
    scale_sdf.exponent = params.scale_exponent;

    // dc = scale * ruv (scaled arithmetic preserves precision)
    ScaledDF dc_re = sdf_mul_scalar(scale_sdf, ruv_x);
    ScaledDF dc_im = sdf_mul_scalar(scale_sdf, ruv_y);

    // Delta z starts at 0 (scaled double-float)
    ScaledDF dz_re;
    dz_re.mantissa = float2(0.0f, 0.0f);
    dz_re.exponent = 0;
    ScaledDF dz_im;
    dz_im.mantissa = float2(0.0f, 0.0f);
    dz_im.exponent = 0;

    // DE derivative tracking (for distance estimation lighting)
    float de_dz_re = 0.0f, de_dz_im = 0.0f;

    // Stripe accumulator
    float stripe_sum = 0.0f, stripe_count = 0.0f, last_stripe = 0.0f;

    int ref_len = params.ref_orbit_len;
    int max_iter = min(params.iter, ref_len - 1);

    float Z_re = 0.0f, Z_im = 0.0f;

    int i = 0;
    for (i = 0; i < max_iter; i++) {
        // X = reference orbit point (stored as double-float in float4)
        float4 X_packed = ref_orbit[i];
        float2 X_re = float2(X_packed.x, X_packed.y);
        float2 X_im = float2(X_packed.z, X_packed.w);

        // Escape check: Z = X + δz
        float dz_re_val = ldexp(dz_re.mantissa.x, dz_re.exponent);
        float dz_im_val = ldexp(dz_im.mantissa.x, dz_im.exponent);
        // Add low component with soft fade
        float low_weight_re = smoothstep(-150.0f, -70.0f, float(dz_re.exponent));
        float low_weight_im = smoothstep(-150.0f, -70.0f, float(dz_im.exponent));
        dz_re_val += ldexp(dz_re.mantissa.y, dz_re.exponent - 23) * low_weight_re;
        dz_im_val += ldexp(dz_im.mantissa.y, dz_im.exponent - 23) * low_weight_im;
        Z_re = X_re.x + dz_re_val;
        Z_im = X_im.x + dz_im_val;

        float mag_sq = Z_re * Z_re + Z_im * Z_im;
        if (mag_sq > 256.0f) break;

        // DE derivative: d/dc(z^2 + c) = 2*z*dz/dc + 1
        if (params.de_lighting == 1) {
            float new_de_dz_re = 2.0f * (Z_re * de_dz_re - Z_im * de_dz_im) + 1.0f;
            float new_de_dz_im = 2.0f * (Z_re * de_dz_im + Z_im * de_dz_re);
            de_dz_re = new_de_dz_re;
            de_dz_im = new_de_dz_im;
        }

        // Stripe average accumulator
        if (params.stripe_intensity > 0.0f) {
            float stripe_val = 0.5f * sin(params.stripe_freq * atan2(Z_im, Z_re)) + 0.5f;
            last_stripe = stripe_val;
            stripe_sum += stripe_val;
            stripe_count += 1.0f;
        }

        // Perturbation formula: δ_{n+1} = 2·X_n·δ_n + δ_n² + δc
        ScaledDF X_dz_re = sdf_sub(sdf_mul_df(dz_re, X_re), sdf_mul_df(dz_im, X_im));
        ScaledDF X_dz_im = sdf_add(sdf_mul_df(dz_re, X_im), sdf_mul_df(dz_im, X_re));
        ScaledDF two_X_dz_re = sdf_mul_scalar(X_dz_re, 2.0f);
        ScaledDF two_X_dz_im = sdf_mul_scalar(X_dz_im, 2.0f);

        ScaledDF dz_sq_re = sdf_sub(sdf_mul(dz_re, dz_re), sdf_mul(dz_im, dz_im));
        ScaledDF dz_sq_im = sdf_mul_scalar(sdf_mul(dz_re, dz_im), 2.0f);

        dz_re = sdf_add(sdf_add(two_X_dz_re, dz_sq_re), dc_re);
        dz_im = sdf_add(sdf_add(two_X_dz_im, dz_sq_im), dc_im);
    }

    // Compute outputs
    float fracIter = -1.0f;  // -1 = inside set
    float dist_est_val = 0.0f;
    float stripe_t = 0.0f;

    if (i < max_iter) {
        // Escaped - compute smooth iteration count
        float4 X_final = ref_orbit[i];
        float dz_re_final = ldexp(dz_re.mantissa.x, dz_re.exponent);
        float dz_im_final = ldexp(dz_im.mantissa.x, dz_im.exponent);
        float low_weight_re_final = smoothstep(-150.0f, -70.0f, float(dz_re.exponent));
        float low_weight_im_final = smoothstep(-150.0f, -70.0f, float(dz_im.exponent));
        dz_re_final += ldexp(dz_re.mantissa.y, dz_re.exponent - 23) * low_weight_re_final;
        dz_im_final += ldexp(dz_im.mantissa.y, dz_im.exponent - 23) * low_weight_im_final;
        Z_re = X_final.x + dz_re_final;
        Z_im = X_final.z + dz_im_final;
        float dist = sqrt(Z_re * Z_re + Z_im * Z_im);

        // Standard smooth iteration
        dist = max(dist, 2.72f);
        float log_dist = log(dist);
        log_dist = max(log_dist, 0.01f);
        fracIter = float(i) + 1.0f - log(log_dist) / log(2.0f);
        fracIter = clamp(fracIter, 0.0f, float(max_iter));

        // Distance estimation value
        if (params.de_lighting == 1) {
            float dz_mag = sqrt(de_dz_re * de_dz_re + de_dz_im * de_dz_im);
            dist_est_val = 2.0f * dist * log(dist) / max(dz_mag, 1e-10f);
        }

        // Stripe interpolation
        if (params.stripe_intensity > 0.0f && stripe_count > 0.0f) {
            float stripe_avg = stripe_sum / stripe_count;
            float prev_avg = (stripe_count > 1.0f) ?
                (stripe_sum - last_stripe) / (stripe_count - 1.0f) : stripe_avg;
            float frac = fract(fracIter);
            stripe_t = mix(prev_avg, stripe_avg, frac);
        }
    }

    frac_buf[idx] = float4(fracIter, dist_est_val, stripe_t, 0.0f);
    de_data[idx] = float4(de_dz_re, de_dz_im, Z_re, Z_im);
}
