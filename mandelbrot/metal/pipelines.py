"""Metal compute pipeline management.

This module handles compilation of Metal shaders and creation of
compute pipelines for the various rendering modes.
"""

import ctypes
import sys
from importlib import resources
from typing import TYPE_CHECKING, Dict

import Metal
import objc

if TYPE_CHECKING:
    from mandelbrot.metal.device import MetalDevice


class PipelineManager:
    """Manages Metal compute pipelines for Mandelbrot rendering.

    This class compiles shaders and creates compute pipeline states for:
    - Perturbation theory (two-pass)
    - Graduated precision direct computation (PREC=1,2,3,4)

    The direct computation pipelines use Metal function constants for
    compile-time specialisation, enabling dead code elimination for
    zero runtime overhead precision selection.

    Attributes:
        pass1_perturb: Pipeline for perturbation pass 1
        pass1_direct: Dictionary mapping precision level (1-4) to pipeline
        pass2: Pipeline for visual effects (shared)
        max_threads: Maximum threads per threadgroup
    """

    def __init__(self, device: 'MetalDevice'):
        """Initialise pipelines by compiling Metal shaders.

        Args:
            device: MetalDevice instance

        Raises:
            SystemExit: If shader compilation fails
        """
        self._device = device
        self._compile_shaders()

    def _load_shader_source(self) -> str:
        """Load and concatenate all shader source files.

        Returns:
            Complete Metal shader source code
        """
        # Load shader files using importlib.resources
        shader_parts = []

        # Note: direct_quad.metal removed - now merged into direct.metal
        shader_files = ['common.metal', 'perturbation.metal', 'direct.metal',
                        'pass2.metal']

        for filename in shader_files:
            try:
                shader_source = resources.files('mandelbrot.metal.shaders').joinpath(filename).read_text()
                shader_parts.append(shader_source)
            except FileNotFoundError:
                print(f"Error: Shader file not found: {filename}")
                sys.exit(1)

        return '\n'.join(shader_parts)

    def _compile_shaders(self):
        """Compile Metal shaders and create pipeline states."""
        metal_source = self._load_shader_source()

        # Compile Metal library
        compile_options = Metal.MTLCompileOptions.new()
        compile_options.setFastMathEnabled_(False)

        library, err = self._device.device.newLibraryWithSource_options_error_(
            metal_source, compile_options, None
        )
        if err is not None:
            print(f"Metal shader compile error:\n{err}", flush=True)
            sys.exit(1)

        # Create pipeline for perturbation pass 1
        fn_pass1 = library.newFunctionWithName_("mandelbrot_perturb_pass1")
        if fn_pass1 is None:
            print("Error: Could not find mandelbrot_perturb_pass1 kernel function")
            sys.exit(1)
        self.pass1_perturb, err = self._device.device.newComputePipelineStateWithFunction_error_(
            fn_pass1, None
        )
        if err is not None:
            print(f"Pipeline pass1 error: {err}")
            sys.exit(1)

        # Create pipeline for pass 2 (shared by all modes)
        fn_pass2 = library.newFunctionWithName_("mandelbrot_perturb_pass2")
        if fn_pass2 is None:
            print("Error: Could not find mandelbrot_perturb_pass2 kernel function")
            sys.exit(1)
        self.pass2, err = self._device.device.newComputePipelineStateWithFunction_error_(
            fn_pass2, None
        )
        if err is not None:
            print(f"Pipeline pass2 error: {err}")
            sys.exit(1)

        # Create specialised pipelines for each precision level (1-4)
        # Uses MTLFunctionConstantValues to set PREC at compile time
        self.pass1_direct: Dict[int, object] = {}

        for prec_level in range(1, 5):
            # Create function constant values
            constant_values = Metal.MTLFunctionConstantValues.alloc().init()

            # Set PREC constant (index 0, type int)
            # Use ctypes to pass the address of the value as required by PyObjC
            prec_val = ctypes.c_int(prec_level)
            constant_values.setConstantValue_type_atIndex_(
                ctypes.addressof(prec_val),
                Metal.MTLDataTypeInt,
                0
            )

            # Create specialised function with constants
            fn_direct, err = library.newFunctionWithName_constantValues_error_(
                "mandelbrot_direct_prec_pass1",
                constant_values,
                None
            )
            if fn_direct is None or err is not None:
                print(f"Error: Could not create specialised function for PREC={prec_level}: {err}")
                sys.exit(1)

            # Create pipeline state
            pipeline, err = self._device.device.newComputePipelineStateWithFunction_error_(
                fn_direct, None
            )
            if err is not None:
                print(f"Pipeline direct PREC={prec_level} error: {err}")
                sys.exit(1)

            self.pass1_direct[prec_level] = pipeline

        # Store max threads for threadgroup calculation
        self.max_threads = self.pass1_perturb.maxTotalThreadsPerThreadgroup()

        # Legacy compatibility attributes (deprecated)
        self.pass1_direct_legacy = self.pass1_direct[2]  # Was "Direct" (double-float)
        self.pass1_direct_quad = self.pass1_direct[4]    # Was "DirectQuad" (quad-float)

    def get_pass1_pipeline(self, render_mode: str):
        """Get the appropriate pass 1 pipeline for the render mode.

        Args:
            render_mode: "Float1", "Float2", "Float3", "Float4", "Direct", "DirectQuad", or "Perturb"

        Returns:
            MTLComputePipelineState for pass 1
        """
        # Handle new precision level modes
        if render_mode.startswith("Float"):
            prec_level = int(render_mode[5:])
            return self.pass1_direct.get(prec_level, self.pass1_perturb)

        # Legacy compatibility
        if render_mode == "Direct":
            return self.pass1_direct[2]  # Double-float
        elif render_mode == "DirectQuad":
            return self.pass1_direct[4]  # Quad-float
        else:
            return self.pass1_perturb

    def get_pass1_pipeline_for_level(self, prec_level: int):
        """Get the appropriate pass 1 pipeline for a precision level.

        Args:
            prec_level: Precision level (1-4) or 0 for perturbation

        Returns:
            MTLComputePipelineState for pass 1
        """
        if prec_level == 0:
            return self.pass1_perturb
        return self.pass1_direct.get(prec_level, self.pass1_perturb)
