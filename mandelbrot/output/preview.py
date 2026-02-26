"""Preview window using Pygame.

This module provides the PreviewWindow class for displaying a live
preview of the render in progress.
"""

from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pygame


class PreviewWindow:
    """Pygame-based live preview window.

    Displays a scaled-down version of rendered frames in real-time
    while encoding is in progress.

    Attributes:
        width: Preview window width
        height: Preview window height
        screen: Pygame display surface
    """

    def __init__(self, render_width: int, render_height: int, scale: float = 0.5):
        """Initialise the preview window.

        Args:
            render_width: Full render width in pixels
            render_height: Full render height in pixels
            scale: Preview scale factor (default 0.5 = half size)
        """
        import pygame
        pygame.init()

        self.width = int(render_width * scale)
        self.height = int(render_height * scale)
        self._scale_step = max(1, int(1 / scale))

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Mandelbrot Render Preview")

        self._pygame = pygame

    def update(self, frame_data: np.ndarray, caption: Optional[str] = None) -> bool:
        """Update the preview with a new frame.

        Args:
            frame_data: Full-resolution RGBA frame data (height, width, 4)
            caption: Optional window caption to display

        Returns:
            True to continue, False if user requested quit
        """
        # Handle events
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                return False
            elif event.type == self._pygame.KEYDOWN:
                if event.key == self._pygame.K_ESCAPE:
                    return False

        # Downsample and display
        preview_array = frame_data[::self._scale_step, ::self._scale_step, :3]
        preview_array = np.ascontiguousarray(preview_array)
        preview_surface = self._pygame.surfarray.make_surface(
            preview_array.swapaxes(0, 1)
        )
        self.screen.blit(preview_surface, (0, 0))
        self._pygame.display.flip()

        if caption:
            self._pygame.display.set_caption(caption)

        return True

    def close(self):
        """Close the preview window."""
        self._pygame.quit()

    def __enter__(self) -> 'PreviewWindow':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close the window."""
        self.close()
        return False
