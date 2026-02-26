"""Metal buffer creation and management.

This module provides functions for creating and managing Metal buffers,
including the reference orbit buffer for perturbation theory.
"""

import struct
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from mpmath import mpf

# Params struct size (aligned for Metal)
# Core params: scale_hi(4) + scale_lo(4) + scale_exponent(4) + ratio(4) + angle(4) + time(4) +
#              zoom_level(4) + colour_freq(4) + lch_lightness(4) + lch_chroma(4) = 40 bytes
# Effect params: emboss_strength(4) + emboss_angle(4) + detail_boost(4) + light_angle_x(4) +
#                light_angle_y(4) + de_ambient(4) + de_diffuse(4) + de_specular(4) +
#                ao_strength(4) + stripe_intensity(4) + stripe_freq(4) + metallic(4) = 48 bytes
# After 88 bytes, need 8 bytes padding to align float3 to 16-byte boundary = 96
# float3 palette_base (16 with padding) + float3 palette_amp (16) + float3 palette_phase (16) = 48 bytes = 144
# int iter, colour_count, ref_orbit_len, width, height, colour_mode, de_lighting = 28 bytes = 172
# 4 bytes padding to 176
PARAMS_SIZE = 176


def mpf_to_double_float(val: 'mpf') -> Tuple[float, float]:
    """Convert mpf to double-float (hi, lo) with maximum precision.

    The hi part captures ~7 decimal digits, and lo captures the next ~7 digits.
    By computing the residual using the EXACT float32 representation (via repr),
    we avoid precision loss from float32→float64 conversion.

    Args:
        val: mpf value from mpmath

    Returns:
        (hi, lo): Two float32 values representing the double-float
    """
    import numpy as np
    from mpmath import mpf

    # Convert to float32 for hi part
    hi = np.float32(float(val))

    # Get exact decimal representation of float32 using repr()
    # This avoids precision loss from float32→float64→mpf conversion
    hi_exact_str = repr(hi.item())
    residual = val - mpf(hi_exact_str)

    # Convert residual to float32 for lo part
    lo = np.float32(float(residual))

    return float(hi), float(lo)


def create_orbit_buffer(device: 'MetalDevice', orbit: List[Tuple['mpf', 'mpf']]):
    """Create Metal buffer containing reference orbit.

    Args:
        device: MetalDevice instance
        orbit: List of (re_mpf, im_mpf) tuples with mpf values

    Returns:
        (buffer, length): Metal buffer and orbit length
    """
    import numpy as np

    # Pack as array of float4 (re_hi, re_lo, im_hi, im_lo) for double-float precision
    # This provides ~14 digits of precision using float32 arithmetic
    # Converting directly from mpf preserves more precision than going through float64

    # Pre-allocate numpy array for faster packing (avoids per-point struct.pack)
    orbit_len = len(orbit)
    orbit_array = np.empty((orbit_len, 4), dtype=np.float32)

    for i, (re, im) in enumerate(orbit):
        re_hi, re_lo = mpf_to_double_float(re)
        im_hi, im_lo = mpf_to_double_float(im)
        orbit_array[i] = (re_hi, re_lo, im_hi, im_lo)

    buffer = device.create_buffer_with_data(orbit_array.tobytes())
    return buffer, orbit_len


def create_render_buffers(device: 'MetalDevice', width: int, height: int):
    """Create all buffers needed for rendering.

    Args:
        device: MetalDevice instance
        width: Frame width in pixels
        height: Frame height in pixels

    Returns:
        Dictionary of buffers: output, frac, de_data, params, params_pass2
    """
    pixel_count = width * height

    return {
        'output': device.create_buffer(pixel_count * 4),      # RGBA output
        'frac': device.create_buffer(pixel_count * 16),       # float4 per pixel
        'de_data': device.create_buffer(pixel_count * 16),    # float4 per pixel
        'params': device.create_buffer(PARAMS_SIZE),
        'params_pass2': device.create_buffer(PARAMS_SIZE),
    }
