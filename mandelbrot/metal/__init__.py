"""Metal GPU abstraction layer.

This module provides a clean interface to Apple's Metal framework for
GPU-accelerated Mandelbrot computation.

Components:
    MetalDevice - Manages Metal device, command queue, and buffers
    PipelineManager - Creates and manages compute pipelines
    create_orbit_buffer - Creates buffer for reference orbit data
"""

from mandelbrot.metal.device import MetalDevice
from mandelbrot.metal.pipelines import PipelineManager
from mandelbrot.metal.buffers import create_orbit_buffer, PARAMS_SIZE

__all__ = ['MetalDevice', 'PipelineManager', 'create_orbit_buffer', 'PARAMS_SIZE']
