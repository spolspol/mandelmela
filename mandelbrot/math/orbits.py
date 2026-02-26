"""Reference orbit computation for perturbation theory.

This module implements the CPU-side arbitrary-precision orbit computation
that enables deep Mandelbrot zooms using perturbation theory.
"""

from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from mpmath import mpf


def compute_reference_orbit(center_re_str: str, center_im_str: str,
                            max_iter: int, precision_digits: int) -> List[Tuple['mpf', 'mpf']]:
    """Compute reference orbit at arbitrary precision using mpmath.

    This is the CPU-side computation that enables deep zooms. We iterate
    z = z² + c at the center point with enough precision to be exact.

    Args:
        center_re_str, center_im_str: Center coordinates as strings
        max_iter: Maximum iterations to compute
        precision_digits: Decimal digits of precision

    Returns:
        List of (X_re_mpf, X_im_mpf) tuples as mpf objects for high-precision conversion
    """
    from mpmath import mp, mpf, mpc

    # Set mpmath precision (digits)
    mp.dps = precision_digits

    # Parse center coordinates at full precision
    c_re = mpf(center_re_str)
    c_im = mpf(center_im_str)
    c = mpc(c_re, c_im)

    # Iterate at arbitrary precision
    z = mpc(0, 0)
    orbit = []

    for i in range(max_iter):
        # Store current point as mpf objects to preserve full precision
        # The conversion to double-float is done in create_orbit_buffer
        # where we can properly extract hi/lo parts from the mpf
        orbit.append((z.real, z.imag))

        # Check escape (use generous bailout for orbit storage)
        z_mag_sq = float(z.real ** 2 + z.imag ** 2)
        if z_mag_sq > 1e16:
            break

        # Iterate: z = z² + c (at full precision)
        z = z * z + c

    return orbit
