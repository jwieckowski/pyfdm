# Copyright (c) 2022 - 2026 Jakub Więckowski

import numpy as np

__all__ = [
    'normalize_weights',
    'crisp_to_tfn_weights',
    'generate_fuzzy_matrix'
]

def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalize fuzzy criteria weights so that all components are in [0, 1].

    If any component of any weight exceeds 1, all weights are scaled by the
    maximum upper bound across all weights. Crisp (1-D) weights are first
    expanded to TFN form (l=m=u for each value).

    Parameters
    ----------
        weights : ndarray, shape (n,) or (n, 3)
            Criteria weights as crisp values or TFNs.

    Returns
    -------
        ndarray, shape (n, 3)
            Normalized TFN weights.

    Raises
    ------
        ValueError – if weights have an unexpected shape.

    Notes
    -----
        This function was previously located in pyfdm.helpers.
        It remains importable from there via a re-export (with a
        DeprecationWarning) for one minor version cycle.
    """
    if weights.ndim == 1:
        weights = np.repeat(weights, 3).reshape((len(weights), 3))
    elif weights.ndim != 2 or weights.shape[1] != 3:
        raise ValueError(
            'Weights must be 1-D (crisp) or 2-D of shape (n, 3) for TFNs, '
            f'got shape {weights.shape}.'
        )

    if np.any(weights > 1):
        max_upper = np.max(weights[:, 2])
        return weights / max_upper

    return weights.copy()


def crisp_to_tfn_weights(weights: np.ndarray) -> np.ndarray:
    """
    Expand crisp weights to degenerate TFNs (l=m=u).

    Parameters
    ----------
        weights : ndarray, shape (n,)
            Crisp weight vector.

    Returns
    -------
        ndarray, shape (n, 3)
            Each weight represented as (w, w, w).
    """
    if weights.ndim != 1:
        raise ValueError(
            f'Expected 1-D crisp weights, got shape {weights.shape}.'
        )
    return np.repeat(weights, 3).reshape((len(weights), 3))

def generate_fuzzy_matrix(
    m: int, 
    n: int,
    lower: float = 0.0, 
    upper: float = 1.0
) -> np.ndarray:
    """
    Generate a random TFN decision matrix.

    Parameters
    ----------
        m : int
            Number of alternatives.
        n : int
            Number of criteria.
        lower : float, default 0.0
            Minimum value of left TFN bound.
        upper : float, default 1.0
            Maximum value of right TFN bound.

    Returns
    -------
        ndarray, shape (m, n, 3)
            Matrix with random TFNs where l <= m <= u.
    """
    if lower > upper:
        raise ValueError('lower must be <= upper.')
        
    matrix = np.random.uniform(low=lower, high=upper, size=(m, n, 3))
    return np.array([np.sort(row, axis=1) for row in matrix])
