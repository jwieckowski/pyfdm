# Copyright (c) 2022 - 2026 Jakub Więckowski

"""
pyfdm.helpers — DEPRECATED.

All functions have moved to pyfdm.utils:

    rank()                  → pyfdm.utils.rank_alternatives()
    normalize_weights()     → pyfdm.utils.normalize_weights()
    generate_fuzzy_matrix() → pyfdm.utils.generate_fuzzy_matrix()

This module re-exports everything for one-version backwards compatibility
and will be removed in v3.0.
"""

import warnings
import numpy as np

from .utils import normalize_weights as _normalize_weights_impl
from .utils import rank_alternatives as _rank_alternatives

__all__ = ['rank', 'generate_fuzzy_matrix', 'normalize_weights']


def rank(x, descending=True):
    """
    Calculate ranking of given values.

    .. deprecated::
        Use ``pyfdm.utils.rank_alternatives()`` instead.
        This function will be removed in v3.0.
    """
    warnings.warn(
        'pyfdm.helpers.rank() is deprecated. '
        'Use: from pyfdm.utils import rank_alternatives',
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        return _rank_alternatives(x, descending=descending, method='average')
    except (TypeError, ValueError) as e:
        raise ValueError(f'Error occurred in ranking calculation: {e}') from e


def generate_fuzzy_matrix(m: int, n: int,
                          lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    """
    Generate a random TFN decision matrix.

    .. deprecated::
        Use ``pyfdm.utils.generate_fuzzy_matrix()`` instead.
        This function will be removed in v3.0.
    """
    warnings.warn(
        'pyfdm.helpers.generate_fuzzy_matrix() is deprecated. '
        'Use: from pyfdm.utils import generate_fuzzy_matrix',
        DeprecationWarning,
        stacklevel=2,
    )
    from .utils import generate_fuzzy_matrix as _gen
    return _gen(m, n, lower, upper)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalize fuzzy criteria weights.

    .. deprecated::
        Use ``pyfdm.utils.normalize_weights()`` instead.
        This function will be removed in v3.0.
    """
    warnings.warn(
        'pyfdm.helpers.normalize_weights() is deprecated. '
        'Use: from pyfdm.utils import normalize_weights',
        DeprecationWarning,
        stacklevel=2,
    )
    return _normalize_weights_impl(weights)
