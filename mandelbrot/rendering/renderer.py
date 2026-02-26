"""Main Mandelbrot renderer orchestrating the complete pipeline.

This module provides the MandelbrotRenderer class which coordinates
all aspects of rendering: Metal setup, frame computation, video encoding,
and optional preview display.
"""

import math
from decimal import Decimal
from typing import Callable, Optional

import numpy as np

from mandelbrot.config import RenderConfig, get_resolution_key
from mandelbrot.metal.device import MetalDevice
from mandelbrot.metal.pipelines import PipelineManager
from mandelbrot.metal.buffers import create_render_buffers
from mandelbrot.math.precision import get_render_mode
from mandelbrot.math.packing import calculate_iterations
from mandelbrot.rendering.frame import FrameComputer
from mandelbrot.rendering.boundary_seeker import BoundarySeeker
from mandelbrot.rendering.region_analysis import find_interesting_regions, pixel_to_complex


class MandelbrotRenderer:
    """Orchestrates the complete Mandelbrot render pipeline.

    This class manages:
    - Metal device and pipeline initialisation
    - Frame-by-frame rendering with adaptive precision
    - Optional boundary seeking for interesting regions
    - Video encoding via FFmpeg
    - Optional preview display

    Attributes:
        config: RenderConfig instance
        device: MetalDevice instance
        pipelines: PipelineManager instance
    """

    def __init__(self, config: RenderConfig):
        """Initialise the renderer.

        Args:
            config: RenderConfig instance with all settings
        """
        self.config = config
        self._resolution_key = get_resolution_key(config.video.width)

        # Initialise Metal
        self.device = MetalDevice()
        self.pipelines = PipelineManager(self.device)

        # Create buffers
        self._buffers = create_render_buffers(
            self.device,
            config.video.width,
            config.video.height
        )

        # Create frame computer
        self._frame_computer = FrameComputer(
            self.device, self.pipelines, config, self._buffers
        )

        # Calculate animation parameters
        self._calculate_animation_params()

        # Initialise boundary seeker if enabled
        self._boundary_seeker = None
        if config.boundary_seek:
            self._init_boundary_seeker()

    def _calculate_animation_params(self):
        """Calculate zoom animation parameters."""
        self.total_frames = max(1, self.config.video.fps * self.config.duration_seconds)
        self.zoom_speed = float(
            (self.config.final_scale / self.config.initial_scale) **
            (Decimal(1) / Decimal(self.total_frames))
        )

    def _init_boundary_seeker(self):
        """Initialise the boundary seeker with appropriate settings."""
        # Calculate frames per 8x magnification for auto seek interval
        frames_per_8x = int(math.log(8) / math.log(1 / self.zoom_speed))
        seek_interval = self.config.seek_interval if self.config.seek_interval > 0 else max(60, frames_per_8x)
        smoothing = max(self.config.seek_smoothing, 30)

        self._boundary_seeker = BoundarySeeker(
            self.config.center_re, self.config.center_im,
            smoothing, seek_interval
        )

    def render_frame(self, frame: int) -> np.ndarray:
        """Render a single frame.

        Args:
            frame: Frame number (0-indexed)

        Returns:
            NumPy array of RGBA pixel data (height, width, 4)
        """
        # Calculate current state
        t = frame / self.config.video.fps
        current_scale = self.config.initial_scale * (Decimal(str(self.zoom_speed)) ** frame)
        current_angle = math.radians(self.config.start_angle) + self.config.rotation_speed * frame

        # Calculate zoom level
        if current_scale > 0:
            zoom_level = float(-current_scale.ln() / Decimal(10).ln())
        else:
            zoom_level = 0
        zoom_level = max(0, zoom_level)

        # Calculate iterations
        iter_count = calculate_iterations(zoom_level, self.config.iter_formula, self.config.base_iter)
        if self.config.max_iter > 0:
            iter_count = min(iter_count, self.config.max_iter)

        # Select render mode
        render_mode = get_render_mode(
            zoom_level, self._resolution_key, self.config.direct_mode
        )

        # Get active center
        center_changed = False
        if self._boundary_seeker:
            self._boundary_seeker.step()
            if render_mode == "Perturb":
                center_re, center_im = self._boundary_seeker.get_stable_center()
            else:
                center_re, center_im = self._boundary_seeker.get_center()
            center_changed = self._boundary_seeker.center_changed()
        else:
            center_re, center_im = self.config.center_re, self.config.center_im

        # Compute frame
        frame_data = self._frame_computer.compute_frame(
            frame, current_scale, current_angle,
            zoom_level, iter_count, render_mode,
            center_re, center_im, center_changed
        )

        # Handle boundary seeking
        if self._boundary_seeker and render_mode != "Perturb":
            self._handle_boundary_seeking(
                frame, current_scale, current_angle,
                center_re, center_im
            )
        elif self._boundary_seeker:
            # In perturbation mode, mark seek done but don't move
            if self._boundary_seeker.should_seek(frame):
                self._boundary_seeker.mark_seek_done(frame)

        # Update orbit cache tracking for boundary seeker
        if self._boundary_seeker and center_changed:
            self._boundary_seeker.mark_orbit_computed(center_re, center_im)

        return frame_data

    def _handle_boundary_seeking(self, frame: int, current_scale: Decimal,
                                 current_angle: float,
                                 center_re: str, center_im: str):
        """Handle XaoS-style boundary seeking.

        Args:
            frame: Current frame number
            current_scale: Current scale
            current_angle: Current rotation angle
            center_re, center_im: Current center coordinates
        """
        if not self._boundary_seeker.should_seek(frame):
            return
        if self._boundary_seeker.should_give_up():
            return

        # Get frac data for region analysis
        frac_data = self._frame_computer.get_frac_data()
        width = self.config.video.width
        height = self.config.video.height

        # Find interesting regions
        regions, overall_variety = find_interesting_regions(
            frac_data, width, height, self.config.seek_grid
        )

        # Mark seek done
        self._boundary_seeker.mark_seek_done(frame)
        self._boundary_seeker.record_variety(overall_variety)

        # Check if we found an interesting region
        if regions and regions[0][2] > self._boundary_seeker.variety_threshold:
            best_x, best_y, _ = regions[0]

            # Calculate direction from centre to interesting region
            cx, cy = width / 2, height / 2
            pixel_dx = best_x - cx
            pixel_dy = best_y - cy

            # Use perpendicular movement to "orbit" the boundary
            perp_dx = -pixel_dy
            perp_dy = pixel_dx

            # Target position: mostly perpendicular + slight inward
            target_px = cx + perp_dx * 0.2 + pixel_dx * 0.05
            target_py = cy + perp_dy * 0.2 + pixel_dy * 0.05

            # Convert to complex coordinates
            new_re, new_im = pixel_to_complex(
                int(target_px), int(target_py), width, height,
                center_re, center_im, current_scale, current_angle
            )

            # Move gently (1% of the way)
            current_re_dec = Decimal(center_re)
            current_im_dec = Decimal(center_im)
            movement_factor = Decimal('0.01')
            target_re = current_re_dec + (Decimal(new_re) - current_re_dec) * movement_factor
            target_im = current_im_dec + (Decimal(new_im) - current_im_dec) * movement_factor
            self._boundary_seeker.update_target(
                str(target_re), str(target_im), frame, perp_dx, perp_dy
            )

        elif self._boundary_seeker.is_getting_boring():
            # Try backtracking
            back_dx, back_dy = self._boundary_seeker.get_backtrack_direction()

            if back_dx != 0 or back_dy != 0:
                cx, cy = width / 2, height / 2
                escape_px = cx + back_dx * width * 0.2
                escape_py = cy + back_dy * height * 0.2
                escape_re, escape_im = pixel_to_complex(
                    int(escape_px), int(escape_py), width, height,
                    center_re, center_im, current_scale, current_angle
                )
                current_re_dec = Decimal(center_re)
                current_im_dec = Decimal(center_im)
                target_re = current_re_dec + (Decimal(escape_re) - current_re_dec) * Decimal('0.01')
                target_im = current_im_dec + (Decimal(escape_im) - current_im_dec) * Decimal('0.01')
                self._boundary_seeker.update_target(
                    str(target_re), str(target_im), frame, back_dx, back_dy
                )
                self._boundary_seeker.trigger_backtrack()

    def render_video(self, output_path: str,
                     progress_callback: Optional[Callable[[int, int, str], None]] = None):
        """Render a complete video to file.

        Args:
            output_path: Path to output video file
            progress_callback: Optional callback(frame, total, mode_str)
        """
        from mandelbrot.output.video import VideoEncoder
        from mandelbrot.math.packing import calculate_iterations

        with VideoEncoder(output_path, self.config.video) as encoder:
            for frame in range(self.total_frames):
                frame_data = self.render_frame(frame)
                encoder.write_frame(frame_data)

                if progress_callback and (frame % 100 == 0 or frame == self.total_frames - 1):
                    # Get render info for callback
                    current_scale = self.config.initial_scale * (Decimal(str(self.zoom_speed)) ** frame)
                    if current_scale > 0:
                        zoom_level = float(-current_scale.ln() / Decimal(10).ln())
                    else:
                        zoom_level = 0
                    iter_count = calculate_iterations(zoom_level, self.config.iter_formula, self.config.base_iter)
                    render_mode = get_render_mode(
                        zoom_level, self._resolution_key, self.config.direct_mode
                    )

                    if render_mode == "Perturb":
                        orbit_len, precision, _ = self._frame_computer.get_cached_orbit_info()
                        mode_str = f"Perturb({precision} digits)"
                    else:
                        mode_str = render_mode

                    progress_callback(frame, self.total_frames, mode_str)

    def render_preview(self, on_frame: Optional[Callable[[np.ndarray, int], bool]] = None):
        """Render with preview display.

        Args:
            on_frame: Optional callback(frame_data, frame) -> should_continue
                     Returns False to stop early
        """
        for frame in range(self.total_frames):
            frame_data = self.render_frame(frame)

            if on_frame:
                if not on_frame(frame_data, frame):
                    break

    def get_frame_info(self, frame: int) -> dict:
        """Get information about a specific frame.

        Args:
            frame: Frame number

        Returns:
            Dictionary with zoom_level, iter_count, render_mode, etc.
        """
        current_scale = self.config.initial_scale * (Decimal(str(self.zoom_speed)) ** frame)
        if current_scale > 0:
            zoom_level = float(-current_scale.ln() / Decimal(10).ln())
        else:
            zoom_level = 0

        iter_count = calculate_iterations(zoom_level, self.config.iter_formula, self.config.base_iter)
        render_mode = get_render_mode(
            zoom_level, self._resolution_key, self.config.direct_mode
        )

        return {
            'frame': frame,
            'zoom_level': zoom_level,
            'iter_count': iter_count,
            'render_mode': render_mode,
            'scale': float(current_scale),
        }
