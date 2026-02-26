#!/usr/bin/env python3
"""Mandelbrot deep-zoom renderer with perturbation theory.

This script is a thin wrapper that invokes the mandelbrot package CLI.
For the full implementation, see the mandelbrot/ package.

Usage:
    python mandelbrot_perturb.py --help
    python mandelbrot_perturb.py -d 5 -m 1e100 -l seahorse -r 720p
    python mandelbrot_perturb.py -d 0.1 -m 1e8 -p -l seahorse -r 360p  # Quick test
"""

from mandelbrot.cli import main

if __name__ == "__main__":
    main()
