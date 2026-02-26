"""Single frame computation logic.

This module contains the FrameComputer class that handles the computation
of individual frames using Metal compute shaders.
"""

import time
from decimal import Decimal
from typing import Optional, Tuple, TYPE_CHECKING

import Metal
import numpy as np

if TYPE_CHECKING:
    from mandelbrot.metal.device import MetalDevice
    from mandelbrot.metal.pipelines import PipelineManager
    from mandelbrot.config import RenderConfig


class FrameComputer:
    """Handles computation of individual Mandelbrot frames.

    This class manages the Metal compute pipeline execution for a single frame,
    including render mode selection, orbit computation, and parameter packing.

    Attributes:
        device: MetalDevice instance
        pipelines: PipelineManager instance
        config: RenderConfig instance
        buffers: Dictionary of Metal buffers
    """

    # Thread group sizes for Metal dispatch
    THREADGROUP_W = 16
    THREADGROUP_H = 16

    def __init__(self, device: 'MetalDevice', pipelines: 'PipelineManager',
                 config: 'RenderConfig', buffers: dict):
        """Initialise the frame computer.

        Args:
            device: MetalDevice instance
            pipelines: PipelineManager instance
            config: RenderConfig instance
            buffers: Dictionary of Metal buffers
        """
        self.device = device
        self.pipelines = pipelines
        self.config = config
        self.buffers = buffers

        self.width = config.video.width
        self.height = config.video.height

        # Cache palette at initialisation (avoids JSON parsing every frame)
        from mandelbrot.data import get_palette
        self._cached_palette = get_palette(
            config.palette,
            config.palette_base,
            config.palette_amp,
            config.palette_phase
        )

        # Reference orbit cache
        self._cached_orbit = None
        self._cached_orbit_buffer = None
        self._cached_precision = 0
        self._cached_center_re = None
        self._cached_center_im = None
        self.last_orbit_compute_time = 0

        # Thread dispatch sizes
        self._threadgroup_size = Metal.MTLSizeMake(
            self.THREADGROUP_W, self.THREADGROUP_H, 1
        )
        self._threads_per_grid = Metal.MTLSizeMake(
            self.width, self.height, 1
        )

    def compute_frame(self, frame: int, current_scale: Decimal, current_angle: float,
                      zoom_level: float, iter_count: int, render_mode: str,
                      center_re: str, center_im: str,
                      center_changed: bool = False) -> np.ndarray:
        """Compute a single frame.

        Args:
            frame: Frame number
            current_scale: Current scale as Decimal
            current_angle: Current rotation angle in radians
            zoom_level: Log10 of magnification
            iter_count: Maximum iterations
            render_mode: "Float1"-"Float4", "Direct", "DirectQuad", or "Perturb"
            center_re, center_im: Center coordinates as strings
            center_changed: Whether center has changed (for orbit recomputation)

        Returns:
            NumPy array of RGBA pixel data (height, width, 4)
        """
        from mandelbrot.math.precision import get_required_precision
        from mandelbrot.math.packing import pack_params, pack_params_direct_prec
        from mandelbrot.math.orbits import compute_reference_orbit
        from mandelbrot.metal.buffers import create_orbit_buffer, PARAMS_SIZE

        t = frame / self.config.video.fps

        # Use cached palette values
        palette_base, palette_amp, palette_phase = self._cached_palette

        # Determine precision level from render mode
        if render_mode.startswith("Float"):
            prec_level = int(render_mode[5:])
        elif render_mode == "Direct":
            prec_level = 2  # Legacy: double-float
        elif render_mode == "DirectQuad":
            prec_level = 4  # Legacy: quad-float
        else:
            prec_level = 0  # Perturbation mode

        if prec_level > 0:
            # Graduated precision direct mode (PREC=1,2,3,4)
            params_data = pack_params_direct_prec(
                current_scale, current_angle, t,
                zoom_level, iter_count,
                prec_level,
                center_re, center_im,
                self.width, self.height,
                self.config.colour_freq, self.config.colour_count,
                self.config.lch_lightness, self.config.lch_chroma,
                self.config.emboss, self.config.emboss_angle,
                self.config.detail_boost,
                self.config.light_angle,
                self.config.ao, self.config.stripe, self.config.stripe_freq,
                self.config.metallic,
                self.config.colour_space, self.config.de_lighting
            )
            orbit_len = 0

        else:
            # Perturbation mode - need reference orbit
            if self.config.precision > 0:
                required_precision = self.config.precision
            else:
                required_precision = get_required_precision(zoom_level)

            # Check if orbit needs recomputation
            # Use +20 threshold to reduce recomputation frequency while maintaining quality
            need_recompute = (
                self._cached_orbit is None or
                required_precision > self._cached_precision + 20 or
                iter_count > len(self._cached_orbit) - 20 or
                center_changed
            )

            if need_recompute:
                orbit_start = time.time()
                orbit_iter = min(iter_count + 100, self.config.max_iter + 100 if self.config.max_iter > 0 else iter_count + 100)
                self._cached_orbit = compute_reference_orbit(
                    center_re, center_im,
                    orbit_iter, required_precision
                )
                self._cached_orbit_buffer, _ = create_orbit_buffer(
                    self.device, self._cached_orbit
                )
                self._cached_precision = required_precision
                self._cached_center_re = center_re
                self._cached_center_im = center_im
                self.last_orbit_compute_time = time.time() - orbit_start

            orbit_len = len(self._cached_orbit)

            params_data = pack_params(
                current_scale, current_angle, t,
                zoom_level, iter_count, orbit_len,
                self.width, self.height,
                self.config.colour_freq, self.config.colour_count,
                self.config.lch_lightness, self.config.lch_chroma,
                self.config.emboss, self.config.emboss_angle,
                self.config.detail_boost,
                self.config.light_angle,
                self.config.ao, self.config.stripe, self.config.stripe_freq,
                self.config.metallic,
                palette_base, palette_amp, palette_phase,
                self.config.colour_space, self.config.de_lighting
            )

        # Copy params to Metal buffer
        self.buffers['params'].contents().as_buffer(PARAMS_SIZE)[:len(params_data)] = params_data

        # Encode and dispatch two-pass rendering
        cmd_buffer = self.device.create_command_buffer()
        encoder = cmd_buffer.computeCommandEncoder()

        # Pass 1: Fractal computation
        if prec_level > 0:
            # Direct computation with graduated precision
            encoder.setComputePipelineState_(self.pipelines.get_pass1_pipeline_for_level(prec_level))
            encoder.setBuffer_offset_atIndex_(self.buffers['params'], 0, 0)
            encoder.setBuffer_offset_atIndex_(self.buffers['frac'], 0, 2)
            encoder.setBuffer_offset_atIndex_(self.buffers['de_data'], 0, 3)
        else:
            # Perturbation mode
            encoder.setComputePipelineState_(self.pipelines.pass1_perturb)
            encoder.setBuffer_offset_atIndex_(self.buffers['params'], 0, 0)
            encoder.setBuffer_offset_atIndex_(self._cached_orbit_buffer, 0, 1)
            encoder.setBuffer_offset_atIndex_(self.buffers['frac'], 0, 2)
            encoder.setBuffer_offset_atIndex_(self.buffers['de_data'], 0, 3)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            self._threads_per_grid, self._threadgroup_size
        )

        # Pass 2: Visual effects and colouring
        encoder.setComputePipelineState_(self.pipelines.pass2)

        # For direct modes, use separate buffer with correct palette values
        if prec_level > 0:
            params_data_pass2 = pack_params(
                current_scale, current_angle, t,
                zoom_level, iter_count, 0,  # orbit_len=0 (not used in pass2)
                self.width, self.height,
                self.config.colour_freq, self.config.colour_count,
                self.config.lch_lightness, self.config.lch_chroma,
                self.config.emboss, self.config.emboss_angle,
                self.config.detail_boost,
                self.config.light_angle,
                self.config.ao, self.config.stripe, self.config.stripe_freq,
                self.config.metallic,
                palette_base, palette_amp, palette_phase,
                self.config.colour_space, self.config.de_lighting
            )
            self.buffers['params_pass2'].contents().as_buffer(PARAMS_SIZE)[:len(params_data_pass2)] = params_data_pass2
            encoder.setBuffer_offset_atIndex_(self.buffers['params_pass2'], 0, 0)
        else:
            encoder.setBuffer_offset_atIndex_(self.buffers['params'], 0, 0)

        encoder.setBuffer_offset_atIndex_(self.buffers['frac'], 0, 1)
        encoder.setBuffer_offset_atIndex_(self.buffers['de_data'], 0, 2)
        encoder.setBuffer_offset_atIndex_(self.buffers['output'], 0, 3)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            self._threads_per_grid, self._threadgroup_size
        )

        encoder.endEncoding()
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()

        # Read output buffer
        pixel_count = self.width * self.height
        data_array = np.frombuffer(
            self.buffers['output'].contents().as_buffer(pixel_count * 4),
            dtype=np.uint8
        ).copy().reshape(self.height, self.width, 4)

        return data_array

    def get_frac_data(self) -> np.ndarray:
        """Get the fractal data buffer for boundary seeking.

        Returns:
            NumPy array of float4 (fracIter, dist_est, stripe_t, 0) per pixel
        """
        pixel_count = self.width * self.height
        return np.frombuffer(
            self.buffers['frac'].contents().as_buffer(pixel_count * 16),
            dtype=np.float32
        ).reshape(pixel_count, 4)

    def get_cached_orbit_info(self) -> Tuple[int, int, float]:
        """Get information about the cached orbit.

        Returns:
            (orbit_len, precision, compute_time)
        """
        orbit_len = len(self._cached_orbit) if self._cached_orbit else 0
        return orbit_len, self._cached_precision, self.last_orbit_compute_time
