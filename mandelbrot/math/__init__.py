"""Mathematical utilities for Mandelbrot computation.

This module provides numerical utilities for precision management,
reference orbit computation, and parameter packing.
"""

from mandelbrot.math.precision import (
    parse_magnification,
    get_render_mode,
    get_required_precision,
    normalize_scale,
)
from mandelbrot.math.orbits import compute_reference_orbit
from mandelbrot.math.packing import (
    pack_params,
    pack_params_direct,
    pack_params_direct_quad,
    calculate_iterations,
)

__all__ = [
    'parse_magnification',
    'get_render_mode',
    'get_required_precision',
    'normalize_scale',
    'compute_reference_orbit',
    'pack_params',
    'pack_params_direct',
    'pack_params_direct_quad',
    'calculate_iterations',
]
