// Common structures and utilities for Mandelbrot shaders
#include <metal_stdlib>
using namespace metal;

struct PerturbParams {
    // Core parameters
    float scale_hi;           // Scale high component (double-float representation)
    float scale_lo;           // Scale low component
    int scale_exponent;       // Power-of-2 exponent: scale = (hi+lo) * 2^exp
    float ratio;              // Aspect ratio
    float angle;              // Rotation angle
    float time;               // Animation time
    float zoom_level;         // Log10 of magnification
    float colour_freq;        // Colour band frequency
    float lch_lightness;      // LCH lightness
    float lch_chroma;         // LCH chroma

    // Visual effect parameters
    float emboss_strength;
    float emboss_angle;
    float detail_boost;
    float light_angle_x;
    float light_angle_y;
    float de_ambient;
    float de_diffuse;
    float de_specular;
    float ao_strength;
    float stripe_intensity;
    float stripe_freq;
    float metallic;

    // Palette (float3 needs 16-byte alignment)
    float3 palette_base;
    float3 palette_amp;
    float3 palette_phase;

    // Integer parameters
    int iter;                 // Max iterations
    int colour_count;         // Number of colour cycles
    int ref_orbit_len;        // Length of reference orbit
    int width;
    int height;
    int colour_mode;          // 0=rgb, 1=lch
    int de_lighting;          // 0=off, 1=on
};

// --- Colour space conversions ---

float3 lch_to_lab(float L, float C, float H) {
    float hr = H * 0.017453293;
    return float3(L, C * cos(hr), C * sin(hr));
}

float3 lab_to_xyz(float3 lab) {
    float fy = (lab.x + 16.0) / 116.0;
    float fx = lab.y / 500.0 + fy;
    float fz = fy - lab.z / 200.0;
    float delta = 6.0 / 29.0;
    float x = (fx > delta) ? fx*fx*fx : (fx - 16.0/116.0) * 3.0 * delta * delta;
    float y = (fy > delta) ? fy*fy*fy : (fy - 16.0/116.0) * 3.0 * delta * delta;
    float z = (fz > delta) ? fz*fz*fz : (fz - 16.0/116.0) * 3.0 * delta * delta;
    return float3(x * 0.95047, y, z * 1.08883);
}

float3 xyz_to_linear_rgb(float3 xyz) {
    return float3(
        dot(xyz, float3( 3.2404542, -1.5371385, -0.4985314)),
        dot(xyz, float3(-0.9692660,  1.8760108,  0.0415560)),
        dot(xyz, float3( 0.0556434, -0.2040259,  1.0572252))
    );
}

float3 linear_to_srgb(float3 c) {
    float3 lo = c * 12.92;
    float3 hi = 1.055 * pow(max(c, float3(0.0)), float3(1.0/2.4)) - 0.055;
    return mix(lo, hi, step(float3(0.0031308), c));
}

float3 lch_to_srgb(float L, float C, float H) {
    float3 lab = lch_to_lab(L, C, H);
    float3 xyz = lab_to_xyz(lab);
    float3 rgb = xyz_to_linear_rgb(xyz);
    return clamp(linear_to_srgb(rgb), 0.0, 1.0);
}

float3 cosine_palette(float t, float3 a, float3 b, float3 c, float3 d) {
    return a + b * cos(6.28318 * (c * t + d));
}

// --- Double-float arithmetic for extended precision ---
// Represents a number as the unevaluated sum of two floats: value = hi + lo
// Provides ~14 decimal digits of precision using only float32 operations

float2 two_sum(float a, float b) {
    float s = a + b;
    float v = s - a;
    float e = (a - (s - v)) + (b - v);
    return float2(s, e);
}

float2 df_split(float a) {
    const float split_const = 4097.0;
    float t = a * split_const;
    float hi = t - (t - a);
    float lo = a - hi;
    return float2(hi, lo);
}

float2 two_prod(float a, float b) {
    float p = a * b;
    float2 sa = df_split(a);
    float2 sb = df_split(b);
    float e = ((sa.x * sb.x - p) + sa.x * sb.y + sa.y * sb.x) + sa.y * sb.y;
    return float2(p, e);
}

// Double-float add: (a.x + a.y) + (b.x + b.y)
float2 df_add(float2 a, float2 b) {
    float2 s = two_sum(a.x, b.x);
    float e = a.y + b.y + s.y;
    return two_sum(s.x, e);
}

// Double-float multiply: (a.x + a.y) * (b.x + b.y)
float2 df_mul(float2 a, float2 b) {
    float2 p = two_prod(a.x, b.x);
    float e = a.x * b.y + a.y * b.x + p.y;
    return two_sum(p.x, e);
}

// Double-float multiply by scalar
float2 df_mul_scalar(float2 a, float s) {
    float2 p = two_prod(a.x, s);
    float e = a.y * s + p.y;
    return two_sum(p.x, e);
}

// --- Triple-float arithmetic (3-component for ~21 decimal digits) ---
// Fills the gap between double-float (14 digits) and quad-float (28 digits).
// Each float32 contributes ~7 decimal digits; 3 components = ~21 digits.
// Faster than quad-float, suitable for zoom 1e10 to 1e16.

void tf_renorm(thread float* a) {
    // Two-pass renormalization for 3 components
    float2 s;

    // First pass (reverse)
    s = two_sum(a[1], a[2]); a[1] = s.x; a[2] = s.y;
    s = two_sum(a[0], a[1]); a[0] = s.x; a[1] = s.y;

    // Second pass (forward)
    s = two_sum(a[0], a[1]); a[0] = s.x; a[1] = s.y;
    s = two_sum(a[1], a[2]); a[1] = s.x; a[2] = s.y;
}

void tf_add(thread float* r, thread const float* a, thread const float* b) {
    // Error-free pairwise addition with error propagation (3 components)
    float t[3], e[3];
    float2 s, s2;

    // Pairwise two_sum
    s = two_sum(a[0], b[0]); t[0] = s.x; e[0] = s.y;
    s = two_sum(a[1], b[1]); t[1] = s.x; e[1] = s.y;
    s = two_sum(a[2], b[2]); t[2] = s.x; e[2] = s.y;

    // Merge errors into next component
    s = two_sum(t[1], e[0]); t[1] = s.x;
    s2 = two_sum(e[1], s.y); e[1] = s2.x;

    s = two_sum(t[2], e[1]); t[2] = s.x;

    r[0] = t[0]; r[1] = t[1]; r[2] = t[2];
    tf_renorm(r);
}

void tf_neg(thread float* r, thread const float* a) {
    for (int i = 0; i < 3; i++) r[i] = -a[i];
}

void tf_mul(thread float* r, thread const float* a, thread const float* b) {
    // Convolution-style multiplication for 3 components
    float t[3] = {0.0f, 0.0f, 0.0f};

    for (int k = 0; k < 3; k++) {
        float sum = t[k];
        float carry_hi = 0.0f;
        float carry_lo = 0.0f;

        for (int i = 0; i <= k; i++) {
            int j = k - i;
            if (j >= 3) continue;

            float2 p = two_prod(a[i], b[j]);
            float2 s = two_sum(sum, p.x);
            sum = s.x;

            float2 ep = two_sum(s.y, p.y);
            float2 c = two_sum(carry_hi, ep.x);
            carry_hi = c.x;
            float2 cl = two_sum(carry_lo, c.y);
            carry_lo = cl.x;
            float2 r1 = two_sum(carry_lo, ep.y);
            carry_lo = r1.x + (cl.y + r1.y);
        }

        t[k] = sum;
        if (k + 1 < 3) {
            float2 s = two_sum(t[k+1], carry_hi);
            t[k+1] = s.x;
            if (k + 2 < 3) {
                float2 s2 = two_sum(t[k+2], carry_lo);
                t[k+2] = s2.x;
            }
        }
    }

    for (int i = 0; i < 3; i++) r[i] = t[i];
    tf_renorm(r);
}

void tf_mul_scalar(thread float* r, thread const float* a, float s) {
    float t[3];
    float carry = 0.0f;

    for (int i = 0; i < 3; i++) {
        float2 p = two_prod(a[i], s);
        float2 c = two_sum(p.x, carry);
        t[i] = c.x;
        float2 e = two_sum(c.y, p.y);
        carry = e.x;
        if (i + 1 < 3) {
            float2 cr = two_sum(carry, e.y);
            carry = cr.x + cr.y;
        }
    }

    for (int i = 0; i < 3; i++) r[i] = t[i];
    tf_renorm(r);
}

// --- Quad-float arithmetic (fixed 4-component for ~28 decimal digits) ---
// Provides precision up to 1e24 zoom without perturbation overhead.
// Each float32 contributes ~7 decimal digits; 4 components = ~28 digits.

void qf_renorm(thread float* a) {
    // Two-pass renormalization: ensures non-overlapping components
    // Unrolled for performance (avoids loop overhead on GPU)
    float2 s;

    // First pass (reverse)
    s = two_sum(a[2], a[3]); a[2] = s.x; a[3] = s.y;
    s = two_sum(a[1], a[2]); a[1] = s.x; a[2] = s.y;
    s = two_sum(a[0], a[1]); a[0] = s.x; a[1] = s.y;

    // Second pass (forward)
    s = two_sum(a[0], a[1]); a[0] = s.x; a[1] = s.y;
    s = two_sum(a[1], a[2]); a[1] = s.x; a[2] = s.y;
    s = two_sum(a[2], a[3]); a[2] = s.x; a[3] = s.y;
}

void qf_add(thread float* r, thread const float* a, thread const float* b) {
    // Error-free pairwise addition with error propagation
    // Unrolled for performance (avoids loop overhead on GPU)
    float t[4], e[4];
    float2 s, s2;

    // Pairwise two_sum (unrolled)
    s = two_sum(a[0], b[0]); t[0] = s.x; e[0] = s.y;
    s = two_sum(a[1], b[1]); t[1] = s.x; e[1] = s.y;
    s = two_sum(a[2], b[2]); t[2] = s.x; e[2] = s.y;
    s = two_sum(a[3], b[3]); t[3] = s.x; e[3] = s.y;

    // Merge errors into next component (unrolled)
    // i=0: merge e[0] into t[1], propagate to e[1]
    s = two_sum(t[1], e[0]); t[1] = s.x;
    s2 = two_sum(e[1], s.y); e[1] = s2.x;

    // i=1: merge e[1] into t[2], propagate to e[2]
    s = two_sum(t[2], e[1]); t[2] = s.x;
    s2 = two_sum(e[2], s.y); e[2] = s2.x;

    // i=2: merge e[2] into t[3] (no further propagation)
    s = two_sum(t[3], e[2]); t[3] = s.x;

    r[0] = t[0]; r[1] = t[1]; r[2] = t[2]; r[3] = t[3];
    qf_renorm(r);
}

void qf_neg(thread float* r, thread const float* a) {
    for (int i = 0; i < 4; i++) r[i] = -a[i];
}

void qf_mul(thread float* r, thread const float* a, thread const float* b) {
    // Convolution-style multiplication with carry propagation
    float t[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k = 0; k < 4; k++) {
        float sum = t[k];
        float carry_hi = 0.0f;
        float carry_lo = 0.0f;

        for (int i = 0; i <= k; i++) {
            int j = k - i;
            if (j >= 4) continue;

            float2 p = two_prod(a[i], b[j]);
            float2 s = two_sum(sum, p.x);
            sum = s.x;

            // Accumulate error into carry chain
            float2 ep = two_sum(s.y, p.y);
            float2 c = two_sum(carry_hi, ep.x);
            carry_hi = c.x;
            float2 cl = two_sum(carry_lo, c.y);
            carry_lo = cl.x;
            float2 r1 = two_sum(carry_lo, ep.y);
            carry_lo = r1.x + (cl.y + r1.y);
        }

        t[k] = sum;
        if (k + 1 < 4) {
            float2 s = two_sum(t[k+1], carry_hi);
            t[k+1] = s.x;
            if (k + 2 < 4) {
                float2 s2 = two_sum(t[k+2], carry_lo);
                t[k+2] = s2.x;
                float2 res = two_sum(s.y, s2.y);
                if (k + 3 < 4) {
                    t[k+3] += res.x + res.y;
                }
            }
        }
    }

    for (int i = 0; i < 4; i++) r[i] = t[i];
    qf_renorm(r);
}

// Multiply quad-float by scalar (for screen coordinate offset)
void qf_mul_scalar(thread float* r, thread const float* a, float s) {
    float t[4];
    float carry = 0.0f;

    for (int i = 0; i < 4; i++) {
        float2 p = two_prod(a[i], s);
        float2 c = two_sum(p.x, carry);
        t[i] = c.x;
        float2 e = two_sum(c.y, p.y);
        carry = e.x;
        if (i + 1 < 4) {
            float2 cr = two_sum(carry, e.y);
            carry = cr.x + cr.y;
        }
    }

    for (int i = 0; i < 4; i++) r[i] = t[i];
    qf_renorm(r);
}

// --- Scaled Double-Float (SDF) for extreme zoom support ---
// Represents value = (mantissa.x + mantissa.y) * 2^exponent
// Keeps mantissa normalized to avoid underflow at deep zooms (1e37+)
// This extends dynamic range from float32's 1e±38 to practically unlimited

struct ScaledDF {
    float2 mantissa;  // double-float (hi, lo), kept in ~[0.5, 2.0) range
    int exponent;     // power-of-2 exponent (can be any int32)
};

// Create scaled value from regular float
ScaledDF sdf_from_float(float val) {
    if (val == 0.0f) {
        ScaledDF result;
        result.mantissa = float2(0.0f, 0.0f);
        result.exponent = 0;
        return result;
    }
    int exp;
    float m = frexp(val, exp);
    ScaledDF result;
    result.mantissa = float2(m, 0.0f);
    result.exponent = exp;
    return result;
}

// Normalize SDF to keep mantissa in a reasonable range
ScaledDF sdf_normalize(ScaledDF v) {
    if (v.mantissa.x == 0.0f && v.mantissa.y == 0.0f) {
        v.exponent = 0;
        return v;
    }

    // Find exponent adjustment using frexp on the high component
    int exp_adjust;
    float normalized = frexp(v.mantissa.x, exp_adjust);

    // Scale mantissa down/up to compensate
    float scale_factor = ldexp(1.0f, -exp_adjust);
    v.mantissa.x *= scale_factor;
    v.mantissa.y *= scale_factor;
    v.exponent += exp_adjust;

    return v;
}

// Add two scaled values (align exponents first)
ScaledDF sdf_add(ScaledDF a, ScaledDF b) {
    // Handle zero cases
    if (a.mantissa.x == 0.0f && a.mantissa.y == 0.0f) return b;
    if (b.mantissa.x == 0.0f && b.mantissa.y == 0.0f) return a;

    int exp_diff = a.exponent - b.exponent;

    // If exponents differ by too much, smaller term is negligible
    if (exp_diff > 126) return a;
    if (exp_diff < -126) return b;

    ScaledDF result;

    if (exp_diff >= 0) {
        result.exponent = a.exponent;
        float scale = ldexp(1.0f, -exp_diff);
        float2 b_scaled = float2(b.mantissa.x * scale, b.mantissa.y * scale);

        // Soft blend zone to minimize artefacts
        if (exp_diff > 44) {
            float blend = 1.0f - smoothstep(44.0f, 126.0f, float(exp_diff));
            b_scaled = float2(b_scaled.x * blend, b_scaled.y * blend);
        }
        result.mantissa = df_add(a.mantissa, b_scaled);
    } else {
        result.exponent = b.exponent;
        float scale = ldexp(1.0f, exp_diff);
        float2 a_scaled = float2(a.mantissa.x * scale, a.mantissa.y * scale);

        if (exp_diff < -44) {
            float blend = 1.0f - smoothstep(44.0f, 126.0f, float(-exp_diff));
            a_scaled = float2(a_scaled.x * blend, a_scaled.y * blend);
        }
        result.mantissa = df_add(a_scaled, b.mantissa);
    }

    return sdf_normalize(result);
}

// Negate a scaled value
ScaledDF sdf_neg(ScaledDF a) {
    a.mantissa = float2(-a.mantissa.x, -a.mantissa.y);
    return a;
}

// Subtract: a - b
ScaledDF sdf_sub(ScaledDF a, ScaledDF b) {
    return sdf_add(a, sdf_neg(b));
}

// Multiply two scaled values (exponents add, mantissas multiply)
ScaledDF sdf_mul(ScaledDF a, ScaledDF b) {
    ScaledDF result;
    result.mantissa = df_mul(a.mantissa, b.mantissa);
    result.exponent = a.exponent + b.exponent;
    return sdf_normalize(result);
}

// Multiply scaled by float scalar (for screen coordinates)
ScaledDF sdf_mul_scalar(ScaledDF a, float s) {
    if (s == 0.0f) {
        ScaledDF result;
        result.mantissa = float2(0.0f, 0.0f);
        result.exponent = 0;
        return result;
    }
    ScaledDF result;
    result.mantissa = df_mul_scalar(a.mantissa, s);
    result.exponent = a.exponent;
    return sdf_normalize(result);
}

// Multiply scaled by regular double-float
ScaledDF sdf_mul_df(ScaledDF a, float2 b) {
    ScaledDF result;
    result.mantissa = df_mul(a.mantissa, b);
    result.exponent = a.exponent;
    return sdf_normalize(result);
}

// Convert SDF to regular float
float sdf_to_float(ScaledDF v) {
    return ldexp(v.mantissa.x, v.exponent);
}
