"""Mandelbrot deep-zoom renderer with perturbation theory.

This package provides GPU-accelerated Mandelbrot set rendering using:
- Metal compute shaders for parallel GPU computation
- Perturbation theory for extreme zoom depths (1e100+)
- Double-float and quad-float arithmetic for extended precision
- XaoS-style boundary seeking for interesting regions

Public API:
    MandelbrotRenderer - Main renderer class
    RenderConfig - Configuration dataclass
    VideoSettings - Video output settings
"""

from mandelbrot.config import RenderConfig, VideoSettings
from mandelbrot.rendering.renderer import MandelbrotRenderer

__all__ = ['MandelbrotRenderer', 'RenderConfig', 'VideoSettings']
__version__ = '1.0.0'
