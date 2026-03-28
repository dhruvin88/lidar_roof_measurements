"""Shared ASPRS LAS classification codes and related constants."""

ASPRS_GROUND: int = 2
ASPRS_BUILDING: int = 6

# Classes that are never roof candidates.
# Used as a fallback exclusion list when class-6 (building) labels are absent.
NON_BUILDING_CLASSES: frozenset[int] = frozenset({
    2,   # Ground
    3,   # Low Vegetation
    4,   # Medium Vegetation
    5,   # High Vegetation
    7,   # Low Point (noise)
    9,   # Water
    10,  # Rail
    11,  # Road Surface
    17,  # Bridge Deck
    18,  # High Noise
    19,  # Overhead Structure
})
