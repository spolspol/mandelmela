"""Output filename generation.

This module provides functions for generating descriptive output filenames
based on render settings.
"""

from typing import Optional


def generate_output_filename(location: str, duration_minutes: float,
                             magnification_log10: float, resolution: str,
                             emboss: float = 0, de_lighting: bool = False,
                             ao: float = 0, stripe: float = 0,
                             metallic: float = 0) -> str:
    """Generate a descriptive output filename.

    The filename encodes the key render parameters for easy identification.

    Args:
        location: Location preset name
        duration_minutes: Video duration in minutes
        magnification_log10: Log10 of final magnification
        resolution: Resolution string (e.g., '720p', '1080p')
        emboss: Emboss strength
        de_lighting: Whether DE lighting is enabled
        ao: Ambient occlusion strength
        stripe: Stripe intensity
        metallic: Metallic effect strength

    Returns:
        Generated filename (e.g., 'mandelbrot_perturb_seahorse_5min_1e100x_720p.mp4')
    """
    mag_exp = int(magnification_log10)
    res_str = resolution.lower()

    # Build effects suffix
    effects = []
    if emboss > 0:
        effects.append(f"emboss{emboss}")
    if de_lighting:
        effects.append("delight")
    if ao > 0:
        effects.append(f"ao{ao}")
    if stripe > 0:
        effects.append(f"stripe{stripe}")
    if metallic > 0:
        effects.append(f"metal{metallic}")

    effects_str = ("_" + "_".join(effects)) if effects else ""

    return (
        f"mandelbrot_perturb_{location}_"
        f"{duration_minutes}min_1e{mag_exp}x_"
        f"{res_str}{effects_str}.mp4"
    )
