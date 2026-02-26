"""Precision management for Mandelbrot computation.

This module handles the complex precision requirements for deep Mandelbrot
zoom, including automatic precision selection and scale normalisation.
"""

import math
from decimal import Decimal, getcontext
from typing import Tuple

import numpy as np

from mandelbrot.config import PRECISION_PROFILES


def parse_magnification(mag_str: str) -> Tuple[float, Decimal]:
    """Parse magnification string and return (log10_value, decimal_value).

    Handles extreme values like '1e1000' that can't be represented as float64.

    Args:
        mag_str: String like '1e12', '1e100', '1e1000', or plain numbers

    Returns:
        (log10_mag, mag_decimal): Log10 of magnification and Decimal representation
    """
    getcontext().prec = 50

    mag_str = mag_str.strip().lower()

    # Handle scientific notation: extract exponent directly
    if 'e' in mag_str:
        parts = mag_str.split('e')
        mantissa = float(parts[0]) if parts[0] else 1.0
        exponent = int(parts[1])
        log10_mag = math.log10(mantissa) + exponent
        # Create Decimal representation for scale calculation
        mag_decimal = Decimal(mantissa) * (Decimal(10) ** exponent)
    else:
        # Plain number (e.g., "1000000")
        mag_decimal = Decimal(mag_str)
        # Use Decimal's log10 for large numbers
        if mag_decimal > 0:
            log10_mag = float(mag_decimal.ln() / Decimal(10).ln())
        else:
            log10_mag = 0
            mag_decimal = Decimal(1)

    return log10_mag, mag_decimal


def get_precision_level(zoom_level: float, resolution_key: str = '4k',
                        direct_mode_enabled: bool = True) -> int:
    """Return appropriate precision level (1-4) or 0 for perturbation mode.

    Uses graduated precision to select the optimal precision level based on
    zoom depth and resolution. Each level uses a different number of float32
    components for extended precision arithmetic:

    - Level 1: float32 (~7 digits), fastest, shallow zooms
    - Level 2: double-float (~14 digits), fast
    - Level 3: triple-float (~21 digits), medium, fills gap between 2 and 4
    - Level 4: quad-float (~28 digits), slower, deep zooms
    - Level 0: Perturbation mode (arbitrary precision via mpmath)

    Args:
        zoom_level: Log10 of current magnification
        resolution_key: Resolution profile key (e.g., '720p', '4k')
        direct_mode_enabled: Whether direct computation modes are enabled

    Returns:
        Precision level 1-4, or 0 for perturbation mode
    """
    # If direct mode disabled, always use perturbation
    if not direct_mode_enabled:
        return 0

    profile = PRECISION_PROFILES.get(resolution_key, PRECISION_PROFILES['4k'])

    # Check thresholds from lowest to highest precision
    if zoom_level < profile[1]:
        return 1
    elif zoom_level < profile[2]:
        return 2
    elif zoom_level < profile[3]:
        return 3
    elif zoom_level < profile[4]:
        return 4
    else:
        return 0  # Perturbation mode


def get_render_mode(zoom_level: float, resolution_key: str = '4k',
                    direct_mode_enabled: bool = True) -> str:
    """Return render mode string for compatibility.

    DEPRECATED: Use get_precision_level() instead for the new graduated
    precision system. This function is kept for backward compatibility.

    Args:
        zoom_level: Log10 of current magnification
        resolution_key: Resolution profile key (e.g., '720p', '4k')
        direct_mode_enabled: Whether direct modes are enabled

    Returns:
        "Float1", "Float2", "Float3", "Float4", or "Perturb"
    """
    level = get_precision_level(zoom_level, resolution_key, direct_mode_enabled)
    if level == 0:
        return "Perturb"
    return f"Float{level}"


def get_required_precision(zoom_level: float) -> int:
    """Calculate required decimal digits of precision for a given zoom level.

    The reference orbit needs enough precision to distinguish the center point
    from its neighbours at the current zoom level, plus margin for error accumulation.

    Args:
        zoom_level: Log10 of current magnification

    Returns:
        Required precision in decimal digits
    """
    # Base precision: zoom level in decimal digits + margin for error accumulation
    # At zoom 1e100, we need ~104 digits; at 1e1000, we need ~1004 digits
    base_precision = int(zoom_level) + 4
    # Minimum precision for shallow zooms
    return max(20, base_precision)


def normalize_scale(scale_decimal: Decimal) -> Tuple[float, float, int]:
    """Normalize scale to double-float mantissa + power-of-2 exponent.

    Returns (scale_hi, scale_lo, exponent) where:
    - scale = (scale_hi + scale_lo) * 2^exponent
    - Mantissa is in range [1.0, 2.0), split into hi/lo for ~14 digit precision

    Args:
        scale_decimal: Scale as Decimal (can be extremely small)

    Returns:
        (scale_hi, scale_lo, exponent): Double-float mantissa and int exponent
    """
    if scale_decimal == 0:
        return 0.0, 0.0, 0

    # Work with absolute value, preserve sign in mantissa
    abs_scale = abs(scale_decimal)

    # Find power-of-2 exponent: log2(scale) = ln(scale) / ln(2)
    ln2 = Decimal(2).ln()
    log2_scale = float(abs_scale.ln() / ln2)
    exponent = int(math.floor(log2_scale))

    # Normalize: mantissa = scale / 2^exponent (gives mantissa in [1.0, 2.0))
    two_power = Decimal(2) ** exponent
    mantissa = scale_decimal / two_power

    # Split into hi + lo components for double-float precision
    # Use repr() to get exact float32 decimal representation for correct residual
    mantissa_hi = np.float32(float(mantissa))
    mantissa_hi_exact = Decimal(repr(mantissa_hi.item()))
    mantissa_lo = np.float32(float(mantissa - mantissa_hi_exact))

    return float(mantissa_hi), float(mantissa_lo), exponent
