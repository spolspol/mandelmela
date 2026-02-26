"""Configuration dataclasses for Mandelbrot renderer.

This module defines the core configuration structures used throughout
the rendering pipeline.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Tuple


@dataclass
class VideoSettings:
    """Video output settings.

    Attributes:
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frames per second
        quality: Encoding quality (0-100)
    """
    width: int
    height: int
    fps: int = 30
    quality: int = 85


@dataclass
class RenderConfig:
    """Complete render configuration.

    Attributes:
        center_re: Real component of zoom centre (string for precision)
        center_im: Imaginary component of zoom centre (string for precision)
        initial_scale: Starting scale (viewport width in complex plane)
        final_scale: Ending scale for zoom animation
        duration_seconds: Total render duration in seconds
        rotation_speed: Rotation speed in radians per frame
        video: Video output settings
        detail_boost: Boost brightness of fine detail areas (0-1)
        metallic: Metallic surface effect strength (0-1)
        boundary_seek: Enable XaoS-style boundary seeking
        seek_interval: Frames between boundary searches
        seek_grid: Grid size for boundary sampling (NxN)
        seek_smoothing: Frames to smooth center transitions
        direct_mode: Enable direct computation for shallow zooms
        preview: Show live preview window
        palette: Colour palette name
        colour_space: Colour space ('rgb' or 'lch')
        colour_freq: Base colour band frequency
        colour_count: Number of colour cycles
        emboss: Emboss effect strength (0-1)
        emboss_angle: Emboss light angle in degrees
        lch_lightness: LCH base lightness (0-100)
        lch_chroma: LCH base chroma (0-130)
        de_lighting: Enable distance estimation lighting
        light_angle: Light direction as (x, y) tuple
        ao: Ambient occlusion strength (0-5)
        stripe: Stripe average intensity (0-1)
        stripe_freq: Stripe frequency
        max_iter: Maximum iterations (0=auto)
        iter_formula: Iteration formula ('log' or 'power')
        base_iter: Minimum iteration count
        precision: Manual precision override (0=auto)
        start_angle: Initial rotation angle in degrees
    """
    # Core zoom parameters
    center_re: str
    center_im: str
    initial_scale: Decimal
    final_scale: Decimal
    duration_seconds: int
    rotation_speed: float
    video: VideoSettings

    # Visual effects
    detail_boost: float = 0.0
    metallic: float = 0.0
    emboss: float = 0.0
    emboss_angle: float = 135.0

    # Boundary seeking
    boundary_seek: bool = False
    seek_interval: int = 0
    seek_grid: int = 32
    seek_smoothing: int = 15

    # Rendering mode
    direct_mode: bool = True
    preview: bool = False

    # Colour settings
    palette: str = 'classic'
    colour_space: str = 'lch'
    colour_freq: float = 0.025
    colour_count: int = 3
    lch_lightness: float = 65.0
    lch_chroma: float = 50.0

    # Palette overrides (None = use preset)
    palette_base: Optional[Tuple[float, float, float]] = None
    palette_amp: Optional[Tuple[float, float, float]] = None
    palette_phase: Optional[Tuple[float, float, float]] = None

    # Lighting
    de_lighting: bool = False
    light_angle: Tuple[float, float] = (-0.5, 0.5)
    ao: float = 0.0
    stripe: float = 0.0
    stripe_freq: float = 3.0

    # Iteration settings
    max_iter: int = 0
    iter_formula: str = 'log'
    base_iter: int = 100
    precision: int = 0
    start_angle: float = 0.0

    def __post_init__(self):
        """Validate and auto-enable dependent settings."""
        # Auto-enable DE lighting when metallic is active
        if self.metallic > 0:
            self.de_lighting = True


# Resolution presets
RESOLUTIONS = {
    '360p': (640, 360),
    '720p': (1280, 720),
    '1080p': (1920, 1080),
    '2k': (2048, 1080),
    '1440p': (2560, 1440),
    '4k': (3840, 2160),
    '8k': (7680, 4320),
}

# Resolution-specific precision thresholds for graduated precision levels.
# Based on: effective_digits = (PREC - 1) × 7, required_digits = zoom + log10(width/3)
#
# PREC=1: float32     ~7 digits   (fastest, shallow zooms)
# PREC=2: double-float ~14 digits (fast)
# PREC=3: triple-float ~21 digits (medium, fills gap between double and quad)
# PREC=4: quad-float   ~28 digits (slower, deep zooms before perturbation)
#
# Each threshold indicates when to switch TO that level (zoom_level <).
# Thresholds pushed to near-maximum for PREC=2 and PREC=4 to delay perturbation.
# At 720p overhead = log10(1280/3) ≈ 2.6, at 4K overhead = log10(3840/3) ≈ 3.1
PRECISION_PROFILES = {
    # Float1 = single float32, Float2 = double-float, Float3 = triple-float
    # Float3/4 limited by float64 input (~16 digits) - transition to perturb at ~1e14
    # This avoids coordinate mismatch glitch (float64 truncation vs mpmath full precision)
    '360p':  {1: 4.5, 2: 11.5, 3: 14.5, 4: 14.5},  # Perturb at 1e14.5
    '720p':  {1: 4.0, 2: 11.0, 3: 14.0, 4: 14.0},  # Perturb at 1e14
    '1080p': {1: 3.8, 2: 10.8, 3: 13.8, 4: 13.8},  # Perturb at 1e13.8
    '2k':    {1: 3.5, 2: 10.5, 3: 13.5, 4: 13.5},  # Perturb at 1e13.5
    '1440p': {1: 3.5, 2: 10.5, 3: 13.5, 4: 13.5},  # Perturb at 1e13.5
    '4k':    {1: 3.2, 2: 10.2, 3: 13.2, 4: 13.2},  # Perturb at 1e13.2
    '8k':    {1: 2.9, 2: 9.9,  3: 12.9, 4: 12.9},  # Perturb at 1e12.9
}


def get_resolution_key(width: int) -> str:
    """Determine resolution profile key based on width.

    Args:
        width: Frame width in pixels

    Returns:
        Resolution profile key for PRECISION_PROFILES
    """
    import math
    overhead = math.log10(width / 3)
    if overhead < 2.9:
        return '720p'
    elif overhead < 3.2:
        return '4k'
    else:
        return '8k'
