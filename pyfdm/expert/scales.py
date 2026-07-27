# Copyright (c) 2026 Jakub Więckowski

"""
Linguistic scales for converting integer expert ratings to
Triangular Fuzzy Numbers (TFNs).

Each scale maps an integer rating to a (l, m, u) tuple.

Scales 1-5 and 1-7 use integer-valued TFN components analogous to
the 1-9 Saaty scale — integer granularity, not float [0,1] compression.
This makes them directly comparable with the 1-9 scale and easier
for experts to interpret.
"""

import numpy as np

__all__ = [
    'SCALE_1_5',
    'SCALE_1_7',
    'SCALE_1_9',
    'get_tfn',
    'list_scales',
]

# ── 1-5 scale (5 linguistic levels, integer-valued TFNs) ────────────────────
#
# Analogous structure to Saaty's 1-9 scale, compressed to 5 levels.
# Each level maps to (l, m, u) with integer components.
#
# Label                   Rating   (l, m, u)
# ─────────────────────── ──────   ─────────
# Equally important            1   (1, 1, 2)
# Slightly more important      2   (1, 2, 3)
# Moderately more important    3   (2, 3, 4)
# Strongly more important      4   (3, 4, 5)
# Absolutely more important    5   (4, 5, 5)

SCALE_1_5 = {
    1: (1, 1, 2),   # Equally important
    2: (1, 2, 3),   # Slightly more important
    3: (2, 3, 4),   # Moderately more important
    4: (3, 4, 5),   # Strongly more important
    5: (4, 5, 5),   # Absolutely more important
}

# ── 1-7 scale (7 linguistic levels, integer-valued TFNs) ────────────────────
#
# Label                        Rating   (l, m, u)
# ──────────────────────────── ──────   ─────────
# Equally important                 1   (1, 1, 2)
# Slightly more important           2   (1, 2, 3)
# Moderately more important         3   (2, 3, 4)
# Moderately to strongly            4   (3, 4, 5)
# Strongly more important           5   (4, 5, 6)
# Very strongly more important      6   (5, 6, 7)
# Absolutely more important         7   (6, 7, 7)

SCALE_1_7 = {
    1: (1, 1, 2),   # Equally important
    2: (1, 2, 3),   # Slightly more important
    3: (2, 3, 4),   # Moderately more important
    4: (3, 4, 5),   # Moderately to strongly
    5: (4, 5, 6),   # Strongly more important
    6: (5, 6, 7),   # Very strongly more important
    7: (6, 7, 7),   # Absolutely more important
}

# ── 1-9 scale (Saaty, 9-point) ───────────────────────────────────────────────
# Commonly used in AHP pairwise comparison matrices.
SCALE_1_9 = {
    1: (1, 1, 2),   # Equally important
    2: (1, 2, 3),   # Between equal and moderate
    3: (2, 3, 4),   # Moderately more important
    4: (3, 4, 5),   # Between moderate and strong
    5: (4, 5, 6),   # Strongly more important
    6: (5, 6, 7),   # Between strong and very strong
    7: (6, 7, 8),   # Very strongly more important
    8: (7, 8, 9),   # Between very strong and absolute
    9: (8, 9, 9),   # Absolutely more important
}

_SCALES = {
    '1-5': SCALE_1_5,
    '1-7': SCALE_1_7,
    '1-9': SCALE_1_9,
}

LV = {
    'Equal': [1, 1, 1],
    'Slightly more important': [1, 2, 3],
    'Weakly important': [2, 3, 4],
    'Between weekly and fairly important': [3, 4, 5],
    'Fairly important': [4, 5, 6],
    'Between fairly and strongly important': [5, 6, 7],
    'Strongly important': [6, 7, 8],
    'Between strongly and absolutely important': [7, 8, 9],
    'Absolutely important': [9, 9, 9]
}


def get_tfn(rating: int, scale: dict) -> np.ndarray:
    """
    Convert an integer rating to a TFN using the given scale.

    Parameters
    ----------
        rating : int
            Integer rating from the expert.

        scale : dict
            Mapping from integer rating to (l, m, u) tuple.
            Use one of SCALE_1_5, SCALE_1_7, SCALE_1_9 or a custom dict.

    Returns
    -------
        ndarray, shape (3,)

    Raises
    ------
        ValueError – if rating is not in the scale.
    """
    if rating not in scale:
        valid = sorted(scale.keys())
        raise ValueError(
            f'Rating {rating} not in scale (valid: {valid}).'
        )
    return np.array(scale[rating], dtype=float)


def list_scales() -> list:
    """Return names of all built-in scales."""
    return list(_SCALES.keys())
