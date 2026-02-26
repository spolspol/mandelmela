"""Command-line interface for Mandelbrot renderer.

This module provides the argument parser and main() entry point for
the Mandelbrot deep-zoom renderer.
"""

import argparse
import math
import sys
from decimal import Decimal

from mandelbrot.config import RenderConfig, VideoSettings, RESOLUTIONS
from mandelbrot.data import get_location, list_locations, get_palette, list_palettes
from mandelbrot.math.precision import parse_magnification
from mandelbrot.output.filenames import generate_output_filename


def parse_rgb(s: str) -> tuple:
    """Parse 'r,g,b' string to tuple of floats."""
    parts = s.split(',')
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: {s}")
    return tuple(float(x.strip()) for x in parts)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all options."""
    parser = argparse.ArgumentParser(
        description='Render Mandelbrot zoom animation using perturbation theory (Metal)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-d', '--duration',
        type=float,
        default=5,
        help='Video duration in minutes'
    )
    parser.add_argument(
        '-m', '--magnification',
        type=str,
        default='1e12',
        help='Final zoom magnification (e.g. 1e100 for 10^100x zoom)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output filename (auto-generated if not specified)'
    )
    parser.add_argument(
        '-p', '--preview',
        action='store_true',
        help='Show live preview window while rendering'
    )
    parser.add_argument(
        '-l', '--location',
        type=str,
        default='seahorse',
        help=f'Zoom location preset. Available: {", ".join(list_locations())}'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=0,
        help='Maximum iterations at end of video (0=auto based on magnification)'
    )
    parser.add_argument(
        '--base-iter',
        type=int,
        default=100,
        help='Base iteration count (minimum iterations, default=100)'
    )
    parser.add_argument(
        '--iter-formula',
        type=str,
        choices=['log', 'power'],
        default='log',
        help='Iteration formula: log=400*ln(scale), power=50*(log10(scale))^1.25'
    )
    parser.add_argument(
        '--colour-freq',
        type=float,
        default=0.025,
        help='Base colour band frequency (higher = more bands)'
    )
    parser.add_argument(
        '--colour-count',
        type=int,
        default=3,
        help='Number of colour cycles in the palette'
    )
    parser.add_argument(
        '-r', '--resolution',
        type=str,
        default='720p',
        help=f'Resolution: {", ".join(RESOLUTIONS.keys())} or WIDTHxHEIGHT'
    )
    parser.add_argument(
        '--palette',
        type=str,
        default='classic',
        choices=list_palettes(),
        help='Colour palette preset'
    )
    parser.add_argument(
        '--precision',
        type=int,
        default=0,
        help='Decimal digits of precision for reference orbit (0=auto based on zoom)'
    )
    parser.add_argument(
        '--start-angle',
        type=float,
        default=0.0,
        help='Initial rotation angle in degrees'
    )
    parser.add_argument(
        '--rotation-speed',
        type=float,
        default=0.003,
        help='Rotation speed in radians per frame (0 to disable)'
    )
    parser.add_argument(
        '--lch-lightness',
        type=float,
        default=65.0,
        help='LCH base lightness (0-100)'
    )
    parser.add_argument(
        '--lch-chroma',
        type=float,
        default=50.0,
        help='LCH base chroma (0-130)'
    )
    parser.add_argument(
        '-e', '--emboss',
        type=float,
        default=0.0,
        help='Emboss strength (0=off, 0.1-1.0 for subtle to full effect)'
    )
    parser.add_argument(
        '--emboss-angle',
        type=float,
        default=135.0,
        help='Emboss light angle in degrees'
    )
    parser.add_argument(
        '--colour-space',
        choices=['rgb', 'lch'],
        default='lch',
        help='Colour space mode'
    )
    parser.add_argument(
        '--palette-base',
        type=str,
        default=None,
        help='Custom base RGB e.g. "0.5,0.3,0.1" (overrides preset)'
    )
    parser.add_argument(
        '--palette-amp',
        type=str,
        default=None,
        help='Custom amplitude RGB e.g. "0.5,0.4,0.3" (overrides preset)'
    )
    parser.add_argument(
        '--palette-phase',
        type=str,
        default=None,
        help='Custom phase RGB e.g. "0.0,0.1,0.2" (overrides preset)'
    )
    parser.add_argument(
        '--de-lighting',
        action='store_true',
        help='Enable distance estimation lighting'
    )
    parser.add_argument(
        '--light-angle',
        type=str,
        default='-0.5,0.5',
        help='Light direction as "x,y"'
    )
    parser.add_argument(
        '--ao',
        type=float,
        default=0.0,
        help='Ambient occlusion strength (0=off, range 0-5)'
    )
    parser.add_argument(
        '--stripe',
        type=float,
        default=0.0,
        help='Stripe average intensity (0=off, range 0-1)'
    )
    parser.add_argument(
        '--stripe-freq',
        type=float,
        default=3.0,
        help='Stripe frequency'
    )
    parser.add_argument(
        '--metallic',
        type=float,
        default=0.0,
        help='Metallic surface strength (0=off, range 0-1)'
    )
    parser.add_argument(
        '--detail-boost',
        type=float,
        default=0.0,
        help='Boost brightness of fine detail areas (0=off, 1=strong)'
    )
    parser.add_argument(
        '--no-direct',
        action='store_true',
        help='Disable direct mode optimisation, always use perturbation theory'
    )
    parser.add_argument(
        '-q', '--quality',
        type=int,
        default=85,
        help='Video quality 0-100 (higher=better)'
    )
    parser.add_argument(
        '--boundary-seek',
        action='store_true',
        help='Automatically seek and zoom toward Mandelbrot set boundary'
    )
    parser.add_argument(
        '--seek-interval',
        type=int,
        default=0,
        help='Frames between boundary searches (0=auto based on zoom speed)'
    )
    parser.add_argument(
        '--seek-grid',
        type=int,
        default=32,
        help='Grid size for boundary sampling (NxN points)'
    )
    parser.add_argument(
        '--seek-smoothing',
        type=int,
        default=15,
        help='Frames to smooth center transitions'
    )

    return parser


def args_to_config(args) -> RenderConfig:
    """Convert parsed arguments to RenderConfig."""
    # Parse magnification
    magnification_log10, magnification_decimal = parse_magnification(args.magnification)

    # Get location
    try:
        location = get_location(args.location)
    except KeyError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Parse resolution
    if args.resolution.lower() in RESOLUTIONS:
        width, height = RESOLUTIONS[args.resolution.lower()]
    elif 'x' in args.resolution:
        try:
            width, height = map(int, args.resolution.lower().split('x'))
        except ValueError:
            print(f"Error: Invalid resolution format '{args.resolution}'")
            sys.exit(1)
    else:
        print(f"Error: Unknown resolution '{args.resolution}'")
        sys.exit(1)

    # Calculate scales
    initial_mag = location.get('start_mag', 300)
    initial_scale = Decimal(3) / Decimal(initial_mag)
    final_scale = Decimal(3) / magnification_decimal

    # Calculate max iterations
    base_iter = args.base_iter
    max_iter = args.max_iter
    if max_iter == 0:
        if args.iter_formula == 'power':
            max_iter = max(base_iter, int(50 * (magnification_log10 ** 1.25)))
        else:
            max_iter = max(base_iter, int(400 * math.log(10 ** magnification_log10)))

    # Parse light angle
    light_parts = args.light_angle.split(',')
    light_angle = (float(light_parts[0].strip()), float(light_parts[1].strip()))

    # Parse palette overrides
    palette_base = parse_rgb(args.palette_base) if args.palette_base else None
    palette_amp = parse_rgb(args.palette_amp) if args.palette_amp else None
    palette_phase = parse_rgb(args.palette_phase) if args.palette_phase else None

    # Create video settings
    video_settings = VideoSettings(
        width=width,
        height=height,
        fps=30,
        quality=args.quality
    )

    # Create render config
    return RenderConfig(
        center_re=location['re'],
        center_im=location['im'],
        initial_scale=initial_scale,
        final_scale=final_scale,
        duration_seconds=max(1, int(args.duration * 60)),
        rotation_speed=args.rotation_speed,
        video=video_settings,
        detail_boost=args.detail_boost,
        metallic=args.metallic,
        emboss=args.emboss,
        emboss_angle=args.emboss_angle,
        boundary_seek=args.boundary_seek,
        seek_interval=args.seek_interval,
        seek_grid=args.seek_grid,
        seek_smoothing=args.seek_smoothing,
        direct_mode=not args.no_direct,
        preview=args.preview,
        palette=args.palette,
        colour_space=args.colour_space,
        colour_freq=args.colour_freq,
        colour_count=args.colour_count,
        lch_lightness=args.lch_lightness,
        lch_chroma=args.lch_chroma,
        palette_base=palette_base,
        palette_amp=palette_amp,
        palette_phase=palette_phase,
        de_lighting=args.de_lighting or args.metallic > 0,
        light_angle=light_angle,
        ao=args.ao,
        stripe=args.stripe,
        stripe_freq=args.stripe_freq,
        max_iter=max_iter,
        iter_formula=args.iter_formula,
        base_iter=base_iter,
        precision=args.precision,
        start_angle=args.start_angle,
    )


def main():
    """Main entry point for the Mandelbrot renderer CLI."""
    # Check for mpmath early
    try:
        from mpmath import mp, mpf, mpc
    except ImportError:
        print("Error: mpmath not installed")
        print("Install with: pip install mpmath")
        sys.exit(1)

    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # Parse magnification for output filename
    magnification_log10, _ = parse_magnification(args.magnification)

    # Get location for output filename
    try:
        location = get_location(args.location)
    except KeyError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Generate output filename if not specified
    output_path = args.output
    if output_path is None:
        output_path = generate_output_filename(
            args.location, args.duration, magnification_log10, args.resolution,
            args.emboss, args.de_lighting or args.metallic > 0,
            args.ao, args.stripe, args.metallic
        )

    # Convert args to config
    config = args_to_config(args)

    # Print settings
    print(f"Video settings: {config.video.width}x{config.video.height} @ {config.video.fps} FPS", flush=True)
    print(f"Location: {location['description']}", flush=True)
    print(f"Duration: {args.duration} min ({config.duration_seconds} seconds)", flush=True)
    print(f"Magnification: {location.get('start_mag', 300):.0f}x -> 1e{magnification_log10:.0f}x", flush=True)
    print(f"Palette: {args.palette}", flush=True)
    print(f"Output: {output_path}", flush=True)

    # Import renderer (triggers Metal setup)
    from mandelbrot.rendering.renderer import MandelbrotRenderer
    from mandelbrot.output.preview import PreviewWindow

    # Create renderer
    print(f"\nInitialising Metal...", flush=True)
    renderer = MandelbrotRenderer(config)
    print(f"Metal Device: {renderer.device.name}", flush=True)
    print(f"Metal pipelines created (max threadgroup: {renderer.pipelines.max_threads})", flush=True)

    if config.direct_mode:
        from mandelbrot.config import PRECISION_PROFILES, get_resolution_key
        res_key = get_resolution_key(config.video.width)
        profile = PRECISION_PROFILES.get(res_key, PRECISION_PROFILES['4k'])
        print(f"Graduated precision enabled: Float1 < 1e{profile[1]}, Float2 < 1e{profile[2]}, "
              f"Float3 < 1e{profile[3]}, Float4 < 1e{profile[4]}", flush=True)
    else:
        print("Direct mode disabled (--no-direct), using perturbation theory for all zoom levels", flush=True)

    if config.boundary_seek:
        print(f"Boundary seeking enabled (XaoS-style): interval={config.seek_interval} frames, "
              f"grid={config.seek_grid}x{config.seek_grid}, smoothing={config.seek_smoothing}", flush=True)

    print(f"\nRendering {renderer.total_frames} frames using perturbation theory...", flush=True)

    # Progress callback
    def progress_callback(frame, total, mode_str):
        info = renderer.get_frame_info(frame)
        print(f"Frame {frame}/{total} ({100*frame/total:.1f}%) "
              f"| Zoom: 1e{info['zoom_level']:.0f} | Iter: {info['iter_count']} | "
              f"Mode: {mode_str}", flush=True)

    # Render with or without preview
    if config.preview:
        preview = PreviewWindow(config.video.width, config.video.height)
        from mandelbrot.output.video import VideoEncoder

        stopped_early = False
        try:
            with VideoEncoder(output_path, config.video) as encoder:
                for frame in range(renderer.total_frames):
                    frame_data = renderer.render_frame(frame)
                    encoder.write_frame(frame_data)

                    # Update preview
                    info = renderer.get_frame_info(frame)
                    if info['render_mode'] == "Perturb":
                        orbit_len, precision, _ = renderer._frame_computer.get_cached_orbit_info()
                        mode_str = f"Perturb({precision})"
                    else:
                        mode_str = info['render_mode']

                    caption = (f"Rendering: {100*frame/renderer.total_frames:.1f}% | "
                               f"Zoom: 1e{info['zoom_level']:.0f} | Iter: {info['iter_count']} | "
                               f"Mode: {mode_str}")
                    if not preview.update(frame_data, caption):
                        stopped_early = True
                        print("\n\nRender stopped by user...", flush=True)
                        break

                    if frame % 100 == 0 or frame == renderer.total_frames - 1:
                        progress_callback(frame, renderer.total_frames, mode_str)

        except KeyboardInterrupt:
            stopped_early = True
            print("\n\nRender interrupted by user.", flush=True)
        finally:
            preview.close()

        if stopped_early:
            print(f"\nRender stopped early: {output_path}")
        else:
            print(f"\nRender complete: {output_path}")
    else:
        try:
            renderer.render_video(output_path, progress_callback)
            print(f"\nRender complete: {output_path}")
        except KeyboardInterrupt:
            print("\n\nRender interrupted by user.")
            sys.exit(1)
        except BrokenPipeError:
            print(f"\nffmpeg pipe broken")
            sys.exit(1)


if __name__ == '__main__':
    main()
