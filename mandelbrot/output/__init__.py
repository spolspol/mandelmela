"""Output handling for video encoding and preview display.

This module provides classes for handling render output:
- VideoEncoder: FFmpeg-based video encoding
- PreviewWindow: Pygame-based live preview
- generate_output_filename: Automatic filename generation
"""

from mandelbrot.output.video import VideoEncoder
from mandelbrot.output.preview import PreviewWindow
from mandelbrot.output.filenames import generate_output_filename

__all__ = ['VideoEncoder', 'PreviewWindow', 'generate_output_filename']
