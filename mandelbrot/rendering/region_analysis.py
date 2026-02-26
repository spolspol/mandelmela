"""Region analysis for boundary seeking.

This module provides functions to analyse rendered frames and find
interesting regions near the Mandelbrot set boundary.
"""

import math
from decimal import Decimal
from typing import List, Tuple

import numpy as np


def find_interesting_regions(frac_data: np.ndarray, width: int, height: int,
                             grid_size: int = 8) -> Tuple[List[Tuple[int, int, float]], float]:
    """Find regions with mix of interior/exterior points (XaoS approach).

    The XaoS autopilot algorithm identifies "interesting" areas as those
    containing BOTH interior (black) AND exterior (coloured) pixels.
    A region with perfect 50/50 mix is maximally interesting.

    Variety score = min(interior_count, exterior_count) / total_count
    - Score 0.0 = all same type (boring - either all black or all coloured)
    - Score 0.5 = perfect 50/50 mix (most interesting - right on boundary)

    Args:
        frac_data: NumPy array of float4 (fracIter, dist_est, stripe_t, 0) per pixel
        width, height: Image dimensions
        grid_size: Divide viewport into NxN regions (default 8 = 64 regions)

    Returns:
        Tuple of:
        - List of (center_x, center_y, variety_score) tuples sorted by score descending
        - Overall viewport variety score
    """
    region_w = width // grid_size
    region_h = height // grid_size

    # Reshape to 2D and extract fracIter values (first component)
    # Sample every 4th pixel using stride slicing
    frac_2d = frac_data[:, 0].reshape(height, width)
    sampled = frac_2d[::4, ::4]  # Every 4th pixel in both dimensions

    # Create interior mask (fracIter < 0 means inside the set)
    interior_mask = sampled < 0

    # Calculate samples per region
    samples_per_region_h = region_h // 4
    samples_per_region_w = region_w // 4

    # Trim to exact multiple of region size for clean reshaping
    trim_h = samples_per_region_h * grid_size
    trim_w = samples_per_region_w * grid_size
    interior_trimmed = interior_mask[:trim_h, :trim_w]

    # Reshape into grid of regions: (grid_size, samples_h, grid_size, samples_w)
    # Then transpose to (grid_size, grid_size, samples_h, samples_w)
    interior_grid = interior_trimmed.reshape(
        grid_size, samples_per_region_h, grid_size, samples_per_region_w
    ).transpose(0, 2, 1, 3)

    # Count interior and exterior per region
    interior_counts = interior_grid.sum(axis=(2, 3))
    total_per_region = samples_per_region_h * samples_per_region_w
    exterior_counts = total_per_region - interior_counts

    # Calculate variety scores: min(interior, exterior) / total
    variety_scores = np.minimum(interior_counts, exterior_counts) / total_per_region

    # Build regions list with center coordinates
    regions = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            cx = gx * region_w + region_w // 2
            cy = gy * region_h + region_h // 2
            regions.append((cx, cy, float(variety_scores[gy, gx])))

    # Sort by variety score (highest = most interesting)
    regions.sort(key=lambda r: r[2], reverse=True)

    # Calculate overall viewport variety
    total_interior = int(interior_counts.sum())
    total_exterior = int(exterior_counts.sum())
    overall_total = total_interior + total_exterior
    if overall_total > 0:
        overall_variety = min(total_interior, total_exterior) / overall_total
    else:
        overall_variety = 0.0

    return regions, overall_variety


def pixel_to_complex(x: int, y: int, width: int, height: int,
                     center_re: str, center_im: str,
                     scale: Decimal, angle: float) -> Tuple[str, str]:
    """Convert pixel coordinates to complex plane coordinates.

    Args:
        x, y: Pixel coordinates
        width, height: Image dimensions
        center_re, center_im: Current center as Decimal strings
        scale: Current scale as Decimal
        angle: Current rotation angle in radians

    Returns:
        (re_str, im_str): Complex coordinates as high-precision strings
    """
    # Normalised screen coordinates [-1, 1]
    ratio = Decimal(width) / Decimal(height)
    nx = (Decimal(2) * Decimal(x) / Decimal(width) - Decimal(1)) * ratio
    ny = Decimal(1) - Decimal(2) * Decimal(y) / Decimal(height)

    # Apply rotation
    cos_a = Decimal(str(math.cos(angle)))
    sin_a = Decimal(str(math.sin(angle)))
    rx = nx * cos_a - ny * sin_a
    ry = nx * sin_a + ny * cos_a

    # Scale and translate to complex plane
    center_re_dec = Decimal(center_re)
    center_im_dec = Decimal(center_im)
    c_re = center_re_dec + rx * scale
    c_im = center_im_dec + ry * scale

    return str(c_re), str(c_im)
