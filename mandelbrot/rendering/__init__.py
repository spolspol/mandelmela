"""Rendering pipeline components.

This module contains the main rendering logic including:
- MandelbrotRenderer: Orchestrates the complete render pipeline
- BoundarySeeker: XaoS-style automatic boundary seeking
- Region analysis: Finding interesting regions for boundary seeking
"""

from mandelbrot.rendering.renderer import MandelbrotRenderer
from mandelbrot.rendering.boundary_seeker import BoundarySeeker
from mandelbrot.rendering.region_analysis import find_interesting_regions, pixel_to_complex

__all__ = [
    'MandelbrotRenderer',
    'BoundarySeeker',
    'find_interesting_regions',
    'pixel_to_complex',
]
