# Copyright (c) 2022 - 2026 Jakub Więckowski

import numpy as np

from ..validator import Validator

__all__ = [
    'equal_weights',
    'shannon_entropy_weights',
    'standard_deviation_weights',
    'variance_weights',
]

def equal_weights(matrix: np.ndarray) -> np.ndarray:
    """
    Assign equal weight to every criterion.

    Parameters
    ----------
    matrix : np.ndarray, shape (m, n, 3)
        TFN decision matrix.

    Returns
    -------
    np.ndarray, shape (n, 3)
        TFN weight vector where each weight equals ``(1/n, 1/n, 1/n)``.

    Raises
    ------
    ValueError
        If `matrix` is not a valid TFN decision matrix (see
        `Validator.validate_matrix_shape`).
    """

    Validator.validate_matrix_shape(matrix)

    try:
        w = np.ones(matrix.shape[1]) / matrix.shape[1]
        return np.repeat(w, 3).reshape((len(w), 3))
    except Exception as e:
        raise RuntimeError(f"Failed to compute equal weights: {e}") from e


def shannon_entropy_weights(matrix: np.ndarray) -> np.ndarray:
    """
    Calculate criterion weights based on Shannon entropy.

    Each column is first normalized into a probability distribution over
    alternatives (``p_ij = x_ij / sum_i(x_ij)``), then its entropy is
    computed on those probabilities. Higher entropy in a column (less
    discriminative criterion, i.e. alternatives are hard to tell apart on
    it) leads to a lower weight. Uses the fuzzy extension of the entropy
    method, applied component-wise to the (l, m, u) parts of the TFN
    matrix.

    Parameters
    ----------
    matrix : np.ndarray, shape (m, n, 3)
        TFN decision matrix. Requires at least 2 alternatives (m >= 2),
        since entropy is undefined (division by ``ln(1) = 0``) for a
        single alternative.

    Returns
    -------
    np.ndarray, shape (n, 3)
        TFN weight vector.

    Raises
    ------
    ValueError
        If `matrix` is not a valid TFN decision matrix (see
        `Validator.validate_matrix_shape`), or has fewer than 2 alternatives.
    """

    Validator.validate_matrix_shape(matrix)

    m, n, _ = matrix.shape

    if m < 2:
        raise ValueError(
            f"'shannon_entropy_weights' requires at least 2 alternatives "
            f"(m >= 2), got m={m}."
        )

    try:
        e = np.zeros((n, 3))
        for j in range(n):
            col = matrix[:, j]

            col_sum = np.sum(col, axis=0)
            col_sum = np.where(col_sum == 0, 1e-10, col_sum)
            p = col / col_sum

            with np.errstate(divide='ignore', invalid='ignore'):
                log_p = np.where(p > 0, np.log(p), 0.0)

            e[j] = -1 / np.log(m) * np.sum(p * log_p, axis=0)

        d = 1 - e
        d_sum = np.sum(d, axis=0)
        d_sum = np.where(d_sum == 0, 1e-10, d_sum)
        w = d / d_sum

    except Exception as e:
        raise RuntimeError(f"Failed to compute Shannon entropy weights: {e}") from e

    return w


def standard_deviation_weights(matrix: np.ndarray) -> np.ndarray:
    """
    Calculate criterion weights based on the standard deviation of each
    column.

    Criteria with more variance across alternatives receive higher
    weights, applied component-wise to the (l, m, u) parts of the TFN
    matrix.

    Parameters
    ----------
    matrix : np.ndarray, shape (m, n, 3)
        TFN decision matrix.

    Returns
    -------
    np.ndarray, shape (n, 3)
        TFN weight vector.

    Raises
    ------
    ValueError
        If `matrix` is not a valid TFN decision matrix (see
        `Validator.validate_matrix_shape`).
    """

    Validator.validate_matrix_shape(matrix)

    try:
        std = np.std(matrix, axis=0)               # (n, 3)
        total = np.sum(std, axis=0)                # (3,)
        total = np.where(total == 0, 1e-10, total)
        return std / total
    except Exception as e:
        raise RuntimeError(f"Failed to compute standard deviation weights: {e}") from e

def variance_weights(matrix: np.ndarray) -> np.ndarray:
    """
    Calculate criterion weights based on the variance of each column.

    Applied component-wise to the (l, m, u) parts of the TFN matrix.

    Parameters
    ----------
    matrix : np.ndarray, shape (m, n, 3)
        TFN decision matrix.

    Returns
    -------
    np.ndarray, shape (n, 3)
        TFN weight vector.

    Raises
    ------
    ValueError
        If `matrix` is not a valid TFN decision matrix (see
        `Validator.validate_matrix_shape`).
    """

    Validator.validate_matrix_shape(matrix)

    try:
        var = np.var(matrix, axis=0)             
        total = np.sum(var, axis=0)
        total = np.where(total == 0, 1e-10, total)
        return var / total
    except Exception as e:
        raise RuntimeError(f"Failed to compute variance weights: {e}") from e
