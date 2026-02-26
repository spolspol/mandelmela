"""Static data files for locations and palettes.

This module provides access to location presets and colour palettes
stored as JSON files.
"""

import json
from importlib import resources
from typing import Dict, Optional, Tuple


def load_locations() -> Dict[str, dict]:
    """Load location presets from JSON file.

    Returns:
        Dictionary mapping location names to their data:
        {name: {'re': str, 'im': str, 'description': str, 'start_mag': int}}
    """
    content = resources.files('mandelbrot.data').joinpath('locations.json').read_text()
    return json.loads(content)


def load_palettes() -> Dict[str, Tuple[Tuple[float, float, float], ...]]:
    """Load colour palettes from JSON file.

    Returns:
        Dictionary mapping palette names to (base, amp, phase) tuples
    """
    content = resources.files('mandelbrot.data').joinpath('palettes.json').read_text()
    data = json.loads(content)
    # Convert lists to tuples
    return {
        name: (tuple(v['base']), tuple(v['amp']), tuple(v['phase']))
        for name, v in data.items()
    }


def get_location(name: str) -> dict:
    """Get a specific location preset.

    Args:
        name: Location name

    Returns:
        Location data dictionary

    Raises:
        KeyError: If location not found
    """
    locations = load_locations()
    if name not in locations:
        available = ', '.join(sorted(locations.keys()))
        raise KeyError(f"Unknown location '{name}'. Available: {available}")
    return locations[name]


def get_palette(name: str,
                base_override: Optional[Tuple[float, float, float]] = None,
                amp_override: Optional[Tuple[float, float, float]] = None,
                phase_override: Optional[Tuple[float, float, float]] = None):
    """Get a colour palette with optional overrides.

    Args:
        name: Palette name
        base_override: Optional override for base RGB
        amp_override: Optional override for amplitude RGB
        phase_override: Optional override for phase RGB

    Returns:
        (base, amp, phase) tuples
    """
    palettes = load_palettes()
    if name not in palettes:
        available = ', '.join(sorted(palettes.keys()))
        raise KeyError(f"Unknown palette '{name}'. Available: {available}")

    base, amp, phase = palettes[name]

    if base_override is not None:
        base = base_override
    if amp_override is not None:
        amp = amp_override
    if phase_override is not None:
        phase = phase_override

    return base, amp, phase


def list_locations() -> list:
    """List all available location names.

    Returns:
        List of location names
    """
    return sorted(load_locations().keys())


def list_palettes() -> list:
    """List all available palette names.

    Returns:
        List of palette names
    """
    return sorted(load_palettes().keys())


# Expose commonly used items at module level
LOCATIONS = None  # Lazily loaded
PALETTES = None   # Lazily loaded


def _ensure_loaded():
    """Ensure data is loaded (for backwards compatibility)."""
    global LOCATIONS, PALETTES
    if LOCATIONS is None:
        LOCATIONS = load_locations()
    if PALETTES is None:
        PALETTES = load_palettes()
