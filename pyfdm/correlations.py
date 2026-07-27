# Copyright (c) 2022 - 2026 Jakub Więckowski

import numpy as np

from .validator import Validator

__all__ = [
    'spearman_coef',
    'pearson_coef',
    'weighted_spearman_coef',
    'ws_rank_similarity_coef'
]

def spearman_coef(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Spearman rank correlation coefficient.

    Parameters
    ----------
    x : np.ndarray
        First ranking vector.

    y : np.ndarray
        Second ranking vector.

    Returns
    -------
    float
        Spearman correlation coefficient.

    Raises
    ------
    ValueError
        If the input vectors are invalid.

    RuntimeError
        If the coefficient cannot be computed.
    """

    x, y = Validator.validate_vectors(x, y)

    try:
        return np.cov(x, y, bias=True)[0, 1] / (np.std(x) * np.std(y))
        
    except Exception as e:
        raise RuntimeError(f"Failed to compute Spearman correlation: {e}") from e

def pearson_coef(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Pearson correlation coefficient.

    Parameters
    ----------
    x : np.ndarray
        First numeric vector.

    y : np.ndarray
        Second numeric vector.

    Returns
    -------
    float
        Pearson correlation coefficient.

    Raises
    ------
    ValueError
        If the input vectors are invalid.

    RuntimeError
        If the coefficient cannot be computed.
    """

    x, y = Validator.validate_vectors(x, y)

    try:
        return np.cov(x, y, bias=True)[0, 1] / (np.std(x) * np.std(y))
        
    except Exception as e:
        raise RuntimeError(f"Failed to compute Pearson correlation: {e}") from e

def weighted_spearman_coef(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Calculate the weighted Spearman rank correlation coefficient.

    Parameters
    ----------
    x : np.ndarray
        First ranking vector.

    y : np.ndarray
        Second ranking vector.

    Returns
    -------
    float
        Weighted Spearman correlation coefficient.

    Raises
    ------
    ValueError
        If the input vectors are invalid.

    RuntimeError
        If the coefficient cannot be computed.
    """

    x, y = Validator.validate_vectors(x, y)

    try:
        n = len(x)
        return 1 - ((6 * np.sum((x-y)**2 * ((n - x + 1) + (n - y + 1)))) / (n**4 + n**3 - n**2 - n))

    except Exception as e:
        raise RuntimeError(f"Failed to compute weighted Spearman correlation: {e}") from e

def ws_rank_similarity_coef(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Calculate the WS rank similarity coefficient.

    Parameters
    ----------
    x : np.ndarray
        First ranking vector.

    y : np.ndarray
        Second ranking vector.

    Returns
    -------
    float
        WS rank similarity coefficient.

    Raises
    ------
    ValueError
        If the input vectors are invalid.

    RuntimeError
        If the coefficient cannot be computed.
    """

    x, y = Validator.validate_vectors(x, y)

    try:

        n = len(x)

        denominator = np.maximum(
            np.abs(1 - x),
            np.abs(n - x),
        )

        denominator = np.where(
            denominator == 0,
            1e-10,
            denominator,
        )

        return 1 - np.sum(2.0 ** (-x) * np.abs(x - y) / denominator)

    except Exception as e:
        raise RuntimeError(f"Failed to compute WS rank similarity coefficient: {e}") from e