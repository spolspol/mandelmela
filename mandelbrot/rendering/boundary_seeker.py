"""XaoS-style boundary seeking for automatic interesting region detection.

This module implements the BoundarySeeker class which automatically guides
the zoom toward the Mandelbrot set boundary for visually interesting renders.
"""

import math
from decimal import Decimal


class BoundarySeeker:
    """Manages automatic boundary seeking with smooth transitions.

    The seeker uses XaoS-style variety detection to identify regions with
    a mix of interior (black) and exterior (coloured) pixels, which
    indicates the boundary of the Mandelbrot set.

    Attributes:
        current_re, current_im: Current render center (Decimal)
        target_re, target_im: Target center we're transitioning to
        smoothing_frames: Number of frames for smooth transitions
        seek_interval: Frames between boundary searches
    """

    def __init__(self, initial_re: str, initial_im: str,
                 smoothing_frames: int, seek_interval: int):
        """Initialise the boundary seeker.

        Args:
            initial_re, initial_im: Starting center coordinates as strings
            smoothing_frames: Number of frames to smooth transitions
            seek_interval: Frames between boundary searches
        """
        # Current center (what we're rendering)
        self.current_re = Decimal(initial_re)
        self.current_im = Decimal(initial_im)

        # Start of transition (stored when new target is set)
        self.start_re = self.current_re
        self.start_im = self.current_im

        # Target center (where we're transitioning to)
        self.target_re = self.current_re
        self.target_im = self.current_im

        # Previous center for orbit cache invalidation
        self.cached_center_re = initial_re
        self.cached_center_im = initial_im

        # Movement direction tracking (in pixel space, normalised)
        self.last_dx = 0.0
        self.last_dy = 0.0

        self.smoothing_frames = smoothing_frames
        self.seek_interval = seek_interval
        self.transition_progress = 1.0  # 1.0 = at target
        self.last_seek_frame = -seek_interval  # Allow immediate first seek

        # XaoS-style variety tracking
        self.variety_history = []  # Recent variety scores
        self.variety_history_max = 5  # How many seeks to remember
        self.variety_threshold = 0.1  # Minimum variety to be "interesting"
        self.boring_trend_length = 3  # Consecutive low-variety seeks before backtrack
        self.backtrack_count = 0  # How many times we've backtracked recently
        self.max_backtrack = 3  # Stop seeking after this many failed backtracks

    def should_seek(self, frame: int) -> bool:
        """Check if we should search for a new boundary point.

        Only seek if:
        1. Enough frames have passed since last seek
        2. Current transition is complete (not mid-movement)

        Args:
            frame: Current frame number

        Returns:
            True if a seek should be performed
        """
        time_ok = frame - self.last_seek_frame >= self.seek_interval
        transition_done = self.transition_progress >= 1.0
        return time_ok and transition_done

    def update_target(self, new_re_str: str, new_im_str: str, frame: int,
                      pixel_dx: float = 0, pixel_dy: float = 0):
        """Set a new target center and begin smooth transition.

        Args:
            new_re_str, new_im_str: New target coordinates as strings
            frame: Current frame number
            pixel_dx, pixel_dy: Direction in pixel space (for consistency tracking)
        """
        # Store current position as start of new transition
        self.start_re = self.current_re
        self.start_im = self.current_im
        # Set new target
        self.target_re = Decimal(new_re_str)
        self.target_im = Decimal(new_im_str)
        self.transition_progress = 0.0
        self.last_seek_frame = frame

        # Update movement direction (normalise)
        mag = math.sqrt(pixel_dx * pixel_dx + pixel_dy * pixel_dy)
        if mag > 0:
            self.last_dx = pixel_dx / mag
            self.last_dy = pixel_dy / mag

    def step(self):
        """Advance the smooth transition by one frame."""
        if self.transition_progress < 1.0:
            # Advance progress
            self.transition_progress += 1.0 / self.smoothing_frames
            self.transition_progress = min(1.0, self.transition_progress)

            # Smooth interpolation using smoothstep (Hermite)
            t = self.transition_progress
            smooth_t = t * t * (3 - 2 * t)

            # Interpolate from start to target
            self.current_re = self.start_re + (self.target_re - self.start_re) * Decimal(str(smooth_t))
            self.current_im = self.start_im + (self.target_im - self.start_im) * Decimal(str(smooth_t))

    def mark_seek_done(self, frame: int):
        """Record that a seek operation was performed at this frame.

        Call this even when no movement occurs to update timing for next seek.

        Args:
            frame: Current frame number
        """
        self.last_seek_frame = frame

    def get_center(self):
        """Return current center coordinates as strings.

        Returns:
            (re_str, im_str): Current center as strings
        """
        return str(self.current_re), str(self.current_im)

    def get_stable_center(self):
        """Return the center that matches the cached orbit.

        In perturbation mode, the rendered center must match the orbit center.
        During transitions, this returns the cached (orbit) center rather than
        the interpolated position to prevent visual flickering.

        Returns:
            (re_str, im_str): Stable center for orbit computation
        """
        if self.transition_progress < 1.0:
            # During transition, return the cached orbit center
            return self.cached_center_re, self.cached_center_im
        else:
            # Transition complete, return current (which should match target)
            return str(self.current_re), str(self.current_im)

    def center_changed(self) -> bool:
        """Check if center has changed since last orbit computation.

        Only returns True when transition is complete to avoid recomputing
        the reference orbit every frame during smooth transitions.
        This prevents visual flickering in perturbation mode.

        Returns:
            True if orbit needs recomputation
        """
        # Don't trigger orbit recomputation during transitions
        if self.transition_progress < 1.0:
            return False

        current_re_str, current_im_str = self.get_center()
        changed = (current_re_str != self.cached_center_re or
                   current_im_str != self.cached_center_im)
        return changed

    def mark_orbit_computed(self, center_re: str = None, center_im: str = None):
        """Mark that an orbit was computed for the given center.

        Args:
            center_re, center_im: The center used for orbit computation.
                If None, uses get_center() (for backwards compatibility).
        """
        if center_re is not None and center_im is not None:
            self.cached_center_re = center_re
            self.cached_center_im = center_im
        else:
            self.cached_center_re, self.cached_center_im = self.get_center()

    def record_variety(self, variety_score: float):
        """Record a variety score from a seek operation.

        Args:
            variety_score: Float 0.0-0.5 indicating mix of interior/exterior
        """
        self.variety_history.append(variety_score)
        # Keep only recent history
        if len(self.variety_history) > self.variety_history_max:
            self.variety_history.pop(0)

        # Reset backtrack counter if we found something interesting
        if variety_score >= self.variety_threshold:
            self.backtrack_count = 0

    def is_getting_boring(self) -> bool:
        """Check if view is becoming boring (declining variety trend).

        Returns True if we've had several consecutive low-variety seeks.
        """
        if len(self.variety_history) < self.boring_trend_length:
            return False

        # Check last N entries
        recent = self.variety_history[-self.boring_trend_length:]
        return all(v < self.variety_threshold for v in recent)

    def should_give_up(self) -> bool:
        """Check if we've tried too many backtracks and should stop seeking.

        Returns:
            True if seeking should be disabled
        """
        return self.backtrack_count >= self.max_backtrack

    def get_backtrack_direction(self):
        """Get the opposite of recent movement direction.

        Returns:
            (dx, dy): Normalised direction opposite to recent travel
        """
        # Simply reverse the last direction
        return -self.last_dx, -self.last_dy

    def trigger_backtrack(self):
        """Called when a backtrack is attempted.

        Increments backtrack counter and clears variety history to give
        the new direction a fresh chance.
        """
        self.backtrack_count += 1
        self.variety_history.clear()
