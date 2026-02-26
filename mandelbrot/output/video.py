"""Video encoding using FFmpeg.

This module provides the VideoEncoder class for encoding rendered frames
into video files using FFmpeg as a subprocess.
"""

import queue
import shutil
import subprocess
import sys
import threading
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from mandelbrot.config import VideoSettings


class VideoEncoder:
    """Context manager for encoding frames to video via FFmpeg.

    Uses FFmpeg's h264_videotoolbox encoder on macOS for hardware-accelerated
    encoding. Frames are piped as raw RGBA data via a background thread
    for improved throughput.

    Example:
        with VideoEncoder('output.mp4', settings) as encoder:
            for frame_data in frames:
                encoder.write_frame(frame_data)
    """

    # Queue size balances memory usage vs pipeline decoupling
    QUEUE_SIZE = 15

    def __init__(self, output_path: str, settings: 'VideoSettings'):
        """Initialise the video encoder.

        Args:
            output_path: Path to output video file
            settings: VideoSettings instance with dimensions, fps, quality

        Raises:
            SystemExit: If ffmpeg is not found in PATH
        """
        self.output_path = output_path
        self.settings = settings
        self._process: Optional[subprocess.Popen] = None
        self._queue: queue.Queue = queue.Queue(maxsize=self.QUEUE_SIZE)
        self._encoder_thread: Optional[threading.Thread] = None
        self._error: Optional[Exception] = None
        self._shutdown = threading.Event()

        # Check for ffmpeg
        if not shutil.which('ffmpeg'):
            print("Error: ffmpeg not found in PATH")
            print("Install with: brew install ffmpeg")
            sys.exit(1)

    def _encoder_worker(self):
        """Background thread that reads frames from queue and writes to FFmpeg."""
        try:
            while not self._shutdown.is_set():
                try:
                    # Use timeout to allow checking shutdown flag
                    frame_data = self._queue.get(timeout=0.1)
                    if frame_data is None:
                        # Sentinel value signals shutdown
                        break
                    self._process.stdin.write(frame_data.tobytes())
                    self._queue.task_done()
                except queue.Empty:
                    continue
        except BrokenPipeError:
            self._error = BrokenPipeError("FFmpeg process terminated unexpectedly")
        except Exception as e:
            self._error = e

    def __enter__(self) -> 'VideoEncoder':
        """Start the FFmpeg process and encoder thread.

        Returns:
            Self for use in with statement
        """
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.settings.width}x{self.settings.height}',
            '-pix_fmt', 'rgba',
            '-r', str(self.settings.fps),
            '-i', '-',
            '-c:v', 'h264_videotoolbox',
            '-q:v', str(self.settings.quality),
            '-pix_fmt', 'yuv420p',
            self.output_path
        ]

        self._process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

        # Start background encoder thread
        self._encoder_thread = threading.Thread(
            target=self._encoder_worker,
            name="FFmpegEncoder",
            daemon=True
        )
        self._encoder_thread.start()

        return self

    def write_frame(self, frame_data: np.ndarray):
        """Queue a frame for encoding (non-blocking unless queue is full).

        Args:
            frame_data: NumPy array of RGBA pixel data (height, width, 4)

        Raises:
            RuntimeError: If encoder not started or background thread errored
        """
        if self._process is None:
            raise RuntimeError("VideoEncoder not started - use as context manager")

        # Check for errors from encoder thread
        if self._error is not None:
            raise self._error

        # Put frame to queue (blocks if queue is full, providing backpressure)
        self._queue.put(frame_data)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Drain queue and close the FFmpeg process.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value
            exc_tb: Exception traceback

        Returns:
            False to propagate any exceptions
        """
        # Signal shutdown and send sentinel
        self._shutdown.set()

        # Drain remaining frames
        if self._encoder_thread is not None:
            # Put sentinel to unblock worker if waiting on empty queue
            try:
                self._queue.put(None, timeout=1.0)
            except queue.Full:
                pass
            self._encoder_thread.join(timeout=30.0)

        if self._process:
            try:
                self._process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            self._process.wait()

            if self._process.returncode != 0 and exc_type is None:
                print(f"Warning: ffmpeg exited with code {self._process.returncode}")

        return False
