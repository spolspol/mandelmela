"""Parameter packing for Metal shaders.

This module provides functions to pack Python parameters into the binary
format expected by the Metal compute shaders.
"""

import math
import struct
from decimal import Decimal
from typing import Tuple

import numpy as np

from mandelbrot.math.precision import normalize_scale

# Pre-compiled struct format for Metal PerturbParams
# Struct layout (aligned for Metal with float3 requiring 16-byte alignment):
# Core: scale_hi(f) + scale_lo(f) + scale_exponent(i) + 19 floats + 8x padding
# Palette: 3x float3 with 4x padding each = 36 bytes
# Integers: 7 ints + 4x padding = 32 bytes
# Total: 176 bytes
_PARAMS_FMT = '=' + 'ffi' + '19f' + '8x' + 'fff4x' + 'fff4x' + 'fff4x' + '7i' + '4x'
_PARAMS_STRUCT = struct.Struct(_PARAMS_FMT)


def calculate_iterations(zoom_level: float, iter_formula: str = 'log',
                         base_iter: int = 100) -> int:
    """Calculate iterations based on selected formula.

    Args:
        zoom_level: Log10 of current magnification
        iter_formula: 'log' or 'power'
        base_iter: Minimum iteration count (default 100)

    Returns:
        Number of iterations to use

    Notes:
        log formula: 400 * ln(scale) - linear scaling with zoom depth
        power formula: 50 * (log10(scale))^1.25 - slower growth at deep zooms
    """
    if zoom_level <= 0:
        return base_iter
    if iter_formula == 'power':
        return max(base_iter, int(50 * (zoom_level ** 1.25)))
    else:
        return max(base_iter, int(400 * math.log(10 ** zoom_level)))


def float_to_multi_float(value: float, n: int) -> list:
    """Convert a float64 value to multi-float representation (n components).

    Each component captures successively finer precision. For n components,
    this provides approximately n × 7 decimal digits of precision.

    Uses Decimal(repr()) to get exact float32 representation, avoiding
    precision loss from float32→float64 conversion in residual calculation.

    Args:
        value: Float64 value to convert
        n: Number of components (1-4)

    Returns:
        List of n float32 values
    """
    from decimal import Decimal

    mf = [0.0] * n
    remaining = Decimal(repr(value))

    for i in range(n):
        hi = np.float32(float(remaining))
        mf[i] = float(hi)
        # Use exact float32 representation for correct residual
        hi_exact = Decimal(repr(hi.item()))
        remaining = remaining - hi_exact

    return mf


def float_to_quad_float(value: float) -> list:
    """Convert a float64 value to quad-float representation (4 components).

    Each component captures successively finer precision, giving ~28 decimal digits total.

    Args:
        value: Float64 value to convert

    Returns:
        List of 4 float32 values
    """
    return float_to_multi_float(value, 4)


def pack_params(scale_decimal: Decimal, angle: float, t: float,
                zoom_level: float, iter_count: int, orbit_len: int,
                width: int, height: int,
                colour_freq: float, colour_count: int,
                lch_lightness: float, lch_chroma: float,
                emboss: float, emboss_angle: float,
                detail_boost: float,
                light_angle: Tuple[float, float],
                ao: float, stripe: float, stripe_freq: float,
                metallic: float,
                palette_base: Tuple[float, float, float],
                palette_amp: Tuple[float, float, float],
                palette_phase: Tuple[float, float, float],
                colour_space: str, de_lighting: bool) -> bytes:
    """Pack parameters into bytes buffer matching Metal struct.

    Args:
        scale_decimal: Current scale as Decimal
        angle: Rotation angle in radians
        t: Animation time
        zoom_level: Log10 of magnification
        iter_count: Maximum iterations
        orbit_len: Length of reference orbit
        width, height: Frame dimensions
        colour_freq: Base colour band frequency
        colour_count: Number of colour cycles
        lch_lightness: LCH base lightness
        lch_chroma: LCH base chroma
        emboss: Emboss strength
        emboss_angle: Emboss light angle in degrees
        detail_boost: Detail boost strength
        light_angle: Light direction as (x, y) tuple
        ao: Ambient occlusion strength
        stripe: Stripe intensity
        stripe_freq: Stripe frequency
        metallic: Metallic effect strength
        palette_base: Palette base RGB
        palette_amp: Palette amplitude RGB
        palette_phase: Palette phase RGB
        colour_space: 'rgb' or 'lch'
        de_lighting: Whether DE lighting is enabled

    Returns:
        Packed bytes matching PerturbParams struct
    """
    # Normalize scale for zoom support (double-float + exponent)
    scale_hi, scale_lo, scale_exponent = normalize_scale(scale_decimal)

    # Use pre-compiled struct for faster packing
    params_data = _PARAMS_STRUCT.pack(
        scale_hi,                 # scale_hi
        scale_lo,                 # scale_lo (for double-float precision)
        scale_exponent,           # scale_exponent
        width / height,           # ratio
        angle,                    # angle
        t,                        # time
        zoom_level,               # zoom_level
        colour_freq,              # colour_freq
        lch_lightness,            # lch_lightness
        lch_chroma,               # lch_chroma
        emboss,                   # emboss_strength
        emboss_angle,             # emboss_angle
        detail_boost,             # detail_boost
        light_angle[0],           # light_angle_x
        light_angle[1],           # light_angle_y
        0.2,                      # de_ambient
        0.6,                      # de_diffuse
        0.2,                      # de_specular
        ao,                       # ao_strength
        stripe,                   # stripe_intensity
        stripe_freq,              # stripe_freq
        metallic,                 # metallic
        palette_base[0], palette_base[1], palette_base[2],    # palette_base
        palette_amp[0], palette_amp[1], palette_amp[2],       # palette_amp
        palette_phase[0], palette_phase[1], palette_phase[2], # palette_phase
        iter_count,               # iter
        colour_count,             # colour_count
        orbit_len,                # ref_orbit_len
        width,                    # width
        height,                   # height
        1 if colour_space == 'lch' else 0,  # colour_mode
        1 if de_lighting else 0,            # de_lighting
    )
    return params_data


def pack_params_direct(scale_decimal: Decimal, angle: float, t: float,
                       zoom_level: float, iter_count: int,
                       center_re_str: str, center_im_str: str,
                       width: int, height: int,
                       colour_freq: float, colour_count: int,
                       lch_lightness: float, lch_chroma: float,
                       emboss: float, emboss_angle: float,
                       detail_boost: float,
                       light_angle: Tuple[float, float],
                       ao: float, stripe: float, stripe_freq: float,
                       metallic: float,
                       palette_amp: Tuple[float, float, float],
                       palette_phase: Tuple[float, float, float],
                       colour_space: str, de_lighting: bool) -> bytes:
    """Pack parameters for direct computation mode (no reference orbit).

    In direct mode, we repurpose the palette_base/amp fields to pass center coordinates
    as double-float values, since the palette isn't used for colouring in pass1.

    Args:
        scale_decimal: Current scale as Decimal
        angle: Rotation angle in radians
        t: Animation time
        zoom_level: Log10 of magnification
        iter_count: Maximum iterations
        center_re_str, center_im_str: Center coordinates as strings
        (remaining args same as pack_params)

    Returns:
        Packed bytes matching PerturbParams struct
    """
    # Normalize scale for GPU (double-float + exponent)
    scale_hi, scale_lo, scale_exponent = normalize_scale(scale_decimal)

    # Convert center coordinates to double-float representation
    center_re = float(center_re_str)
    center_im = float(center_im_str)
    center_re_hi = np.float32(center_re)
    center_re_lo = np.float32(center_re - float(center_re_hi))
    center_im_hi = np.float32(center_im)
    center_im_lo = np.float32(center_im - float(center_im_hi))

    # Use pre-compiled struct for faster packing
    params_data = _PARAMS_STRUCT.pack(
        scale_hi,                 # scale_hi
        scale_lo,                 # scale_lo
        scale_exponent,           # scale_exponent
        width / height,           # ratio
        angle,                    # angle
        t,                        # time
        zoom_level,               # zoom_level
        colour_freq,              # colour_freq
        lch_lightness,            # lch_lightness
        lch_chroma,               # lch_chroma
        emboss,                   # emboss_strength
        emboss_angle,             # emboss_angle
        detail_boost,             # detail_boost
        light_angle[0],           # light_angle_x
        light_angle[1],           # light_angle_y
        0.2,                      # de_ambient
        0.6,                      # de_diffuse
        0.2,                      # de_specular
        ao,                       # ao_strength
        stripe,                   # stripe_intensity
        stripe_freq,              # stripe_freq
        metallic,                 # metallic
        # Center coords packed into palette_base (repurposed for direct mode)
        float(center_re_hi), float(center_re_lo), float(center_im_hi),  # palette_base
        float(center_im_lo), palette_amp[1], palette_amp[2],            # palette_amp
        palette_phase[0], palette_phase[1], palette_phase[2],           # palette_phase
        iter_count,               # iter
        colour_count,             # colour_count
        0,                        # ref_orbit_len (not used in direct mode)
        width,                    # width
        height,                   # height
        1 if colour_space == 'lch' else 0,  # colour_mode
        1 if de_lighting else 0,            # de_lighting
    )
    return params_data


def pack_params_direct_quad(scale_decimal: Decimal, angle: float, t: float,
                            zoom_level: float, iter_count: int,
                            center_re_str: str, center_im_str: str,
                            width: int, height: int,
                            colour_freq: float, colour_count: int,
                            lch_lightness: float, lch_chroma: float,
                            emboss: float, emboss_angle: float,
                            detail_boost: float,
                            light_angle: Tuple[float, float],
                            ao: float, stripe: float, stripe_freq: float,
                            metallic: float,
                            palette_phase: Tuple[float, float, float],
                            colour_space: str, de_lighting: bool) -> bytes:
    """Pack parameters for quad-float direct computation mode (no reference orbit).

    DEPRECATED: Use pack_params_direct_prec() with prec_level=4 instead.

    Quad-float provides ~28 decimal digits, enabling zoom up to 1e24.

    Center coordinates packed via palette fields:
      palette_base.xyz = center_re[0:3]
      palette_amp.x    = center_re[3]
      palette_amp.yz   = center_im[0:2]
      palette_phase.xy = center_im[2:4]

    Args:
        (same as pack_params_direct, except palette_amp not used)

    Returns:
        Packed bytes matching PerturbParams struct
    """
    # Normalize scale for GPU (double-float + exponent)
    scale_hi, scale_lo, scale_exponent = normalize_scale(scale_decimal)

    # Convert center coordinates to quad-float representation
    center_re = float(center_re_str)
    center_im = float(center_im_str)
    qf_re = float_to_quad_float(center_re)
    qf_im = float_to_quad_float(center_im)

    # Use pre-compiled struct for faster packing
    params_data = _PARAMS_STRUCT.pack(
        scale_hi,                 # scale_hi
        scale_lo,                 # scale_lo
        scale_exponent,           # scale_exponent
        width / height,           # ratio
        angle,                    # angle
        t,                        # time
        zoom_level,               # zoom_level
        colour_freq,              # colour_freq
        lch_lightness,            # lch_lightness
        lch_chroma,               # lch_chroma
        emboss,                   # emboss_strength
        emboss_angle,             # emboss_angle
        detail_boost,             # detail_boost
        light_angle[0],           # light_angle_x
        light_angle[1],           # light_angle_y
        0.2,                      # de_ambient
        0.6,                      # de_diffuse
        0.2,                      # de_specular
        ao,                       # ao_strength
        stripe,                   # stripe_intensity
        stripe_freq,              # stripe_freq
        metallic,                 # metallic
        # Quad-float center coords packed into palette fields
        qf_re[0], qf_re[1], qf_re[2],  # palette_base = center_re[0:3]
        qf_re[3], qf_im[0], qf_im[1],  # palette_amp = (center_re[3], center_im[0:2])
        qf_im[2], qf_im[3], palette_phase[2],  # palette_phase = (center_im[2:4], unused)
        iter_count,               # iter
        colour_count,             # colour_count
        0,                        # ref_orbit_len (not used in direct mode)
        width,                    # width
        height,                   # height
        1 if colour_space == 'lch' else 0,  # colour_mode
        1 if de_lighting else 0,            # de_lighting
    )
    return params_data


def pack_params_direct_prec(scale_decimal: Decimal, angle: float, t: float,
                            zoom_level: float, iter_count: int,
                            prec_level: int,
                            center_re_str: str, center_im_str: str,
                            width: int, height: int,
                            colour_freq: float, colour_count: int,
                            lch_lightness: float, lch_chroma: float,
                            emboss: float, emboss_angle: float,
                            detail_boost: float,
                            light_angle: Tuple[float, float],
                            ao: float, stripe: float, stripe_freq: float,
                            metallic: float,
                            colour_space: str, de_lighting: bool) -> bytes:
    """Pack parameters for graduated precision direct computation mode.

    Supports precision levels 1-4 with different coordinate packing:
      PREC=1: palette_base.xy = (center_re, center_im)
      PREC=2: palette_base = (re_hi, re_lo, im_hi), palette_amp.x = im_lo
      PREC=3: palette_base = re[0:3], palette_amp.xyz = im[0:3]
      PREC=4: palette_base.xyz = re[0:3], palette_amp = (re[3], im[0:2]), palette_phase.xy = im[2:4]

    Args:
        scale_decimal: Current scale as Decimal
        angle: Rotation angle in radians
        t: Animation time
        zoom_level: Log10 of magnification
        iter_count: Maximum iterations
        prec_level: Precision level (1-4)
        center_re_str, center_im_str: Center coordinates as strings
        width, height: Frame dimensions
        colour_freq: Base colour band frequency
        colour_count: Number of colour cycles
        lch_lightness: LCH base lightness
        lch_chroma: LCH base chroma
        emboss: Emboss strength
        emboss_angle: Emboss light angle in degrees
        detail_boost: Detail boost strength
        light_angle: Light direction as (x, y) tuple
        ao: Ambient occlusion strength
        stripe: Stripe intensity
        stripe_freq: Stripe frequency
        metallic: Metallic effect strength
        colour_space: 'rgb' or 'lch'
        de_lighting: Whether DE lighting is enabled

    Returns:
        Packed bytes matching PerturbParams struct
    """
    # Normalize scale for GPU (double-float + exponent)
    scale_hi, scale_lo, scale_exponent = normalize_scale(scale_decimal)

    # Convert center coordinates to float
    center_re = float(center_re_str)
    center_im = float(center_im_str)

    # Pack coordinates based on precision level
    if prec_level == 1:
        # PREC=1: Simple float32 - palette_base.xy = (center_re, center_im)
        pb0 = center_re
        pb1 = center_im
        pb2 = 0.0
        pa0 = 0.0
        pa1 = 0.0
        pa2 = 0.0
        pp0 = 0.0
        pp1 = 0.0
        pp2 = 0.0

    elif prec_level == 2:
        # PREC=2: Double-float - palette_base = (re_hi, re_lo, im_hi), palette_amp.x = im_lo
        mf_re = float_to_multi_float(center_re, 2)
        mf_im = float_to_multi_float(center_im, 2)
        pb0 = mf_re[0]
        pb1 = mf_re[1]
        pb2 = mf_im[0]
        pa0 = mf_im[1]
        pa1 = 0.0
        pa2 = 0.0
        pp0 = 0.0
        pp1 = 0.0
        pp2 = 0.0

    elif prec_level == 3:
        # PREC=3: Triple-float - palette_base = re[0:3], palette_amp = im[0:3]
        mf_re = float_to_multi_float(center_re, 3)
        mf_im = float_to_multi_float(center_im, 3)
        pb0 = mf_re[0]
        pb1 = mf_re[1]
        pb2 = mf_re[2]
        pa0 = mf_im[0]
        pa1 = mf_im[1]
        pa2 = mf_im[2]
        pp0 = 0.0
        pp1 = 0.0
        pp2 = 0.0

    else:  # prec_level == 4
        # PREC=4: Quad-float - palette_base.xyz = re[0:3], palette_amp = (re[3], im[0:2]), palette_phase.xy = im[2:4]
        mf_re = float_to_multi_float(center_re, 4)
        mf_im = float_to_multi_float(center_im, 4)
        pb0 = mf_re[0]
        pb1 = mf_re[1]
        pb2 = mf_re[2]
        pa0 = mf_re[3]
        pa1 = mf_im[0]
        pa2 = mf_im[1]
        pp0 = mf_im[2]
        pp1 = mf_im[3]
        pp2 = 0.0

    # Use pre-compiled struct for faster packing
    params_data = _PARAMS_STRUCT.pack(
        scale_hi,                 # scale_hi
        scale_lo,                 # scale_lo
        scale_exponent,           # scale_exponent
        width / height,           # ratio
        angle,                    # angle
        t,                        # time
        zoom_level,               # zoom_level
        colour_freq,              # colour_freq
        lch_lightness,            # lch_lightness
        lch_chroma,               # lch_chroma
        emboss,                   # emboss_strength
        emboss_angle,             # emboss_angle
        detail_boost,             # detail_boost
        light_angle[0],           # light_angle_x
        light_angle[1],           # light_angle_y
        0.2,                      # de_ambient
        0.6,                      # de_diffuse
        0.2,                      # de_specular
        ao,                       # ao_strength
        stripe,                   # stripe_intensity
        stripe_freq,              # stripe_freq
        metallic,                 # metallic
        pb0, pb1, pb2,            # palette_base (center coords)
        pa0, pa1, pa2,            # palette_amp (center coords continued)
        pp0, pp1, pp2,            # palette_phase (center coords continued for PREC=4)
        iter_count,               # iter
        colour_count,             # colour_count
        0,                        # ref_orbit_len (not used in direct mode)
        width,                    # width
        height,                   # height
        1 if colour_space == 'lch' else 0,  # colour_mode
        1 if de_lighting else 0,            # de_lighting
    )
    return params_data
