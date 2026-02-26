"""Metal device management.

This module handles Metal device initialisation and basic buffer operations.
"""

import sys

import Metal


class MetalDevice:
    """Manages Metal device and command queue.

    This class encapsulates the Metal device setup and provides methods
    for creating buffers and command queues.

    Attributes:
        device: The underlying MTLDevice
        command_queue: Command queue for dispatching work
        name: Device name string
    """

    def __init__(self):
        """Initialise the Metal device.

        Raises:
            SystemExit: If no Metal device is found
        """
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            print("Error: No Metal device found")
            sys.exit(1)

        self.command_queue = self.device.newCommandQueue()
        self.name = self.device.name()

    def create_buffer(self, size: int) -> 'Metal.MTLBuffer':
        """Create a shared-storage Metal buffer.

        Args:
            size: Buffer size in bytes

        Returns:
            Metal buffer with shared storage mode
        """
        return self.device.newBufferWithLength_options_(
            size, Metal.MTLResourceStorageModeShared
        )

    def create_buffer_with_data(self, data: bytes) -> 'Metal.MTLBuffer':
        """Create a Metal buffer initialised with data.

        Args:
            data: Bytes to copy into the buffer

        Returns:
            Metal buffer containing the data
        """
        return self.device.newBufferWithBytes_length_options_(
            data, len(data), Metal.MTLResourceStorageModeShared
        )

    def create_command_buffer(self):
        """Create a new command buffer.

        Returns:
            MTLCommandBuffer ready for encoding
        """
        return self.command_queue.commandBuffer()
