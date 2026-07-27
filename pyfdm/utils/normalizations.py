# Copyright (c) 2022 - 2026 Jakub Więckowski

import numpy as np

from ..validator import Validator

__all__ = [
    'cocoso_normalization',
    'linear_normalization',
    'max_normalization',
    'minmax_normalization',
    'saw_normalization',
    'sum_normalization',
    'sqrt_normalization',
    'vector_normalization',
    'waspas_normalization'
]

def sum_normalization(
    matrix: np.ndarray | list,
    types: np.ndarray | list
) -> np.ndarray:
    """
    Normalize a Triangular Fuzzy Number decision matrix using sum
    normalization.

    Benefit criteria are normalized by the sum of fuzzy values, while cost
    criteria are transformed using reciprocal values.

    Parameters
    ----------
    matrix : np.ndarray | list
        TFN decision matrix with shape (m, n, 3).

    types : np.ndarray | list
        Criterion optimization directions:
        1 for benefit criteria,
        -1 for cost criteria.

    Returns
    -------
    np.ndarray
        Normalized TFN decision matrix with shape (m, n, 3).

    Raises
    ------
    ValueError
        If the matrix or criterion types are invalid.

    RuntimeError
        If normalization fails.
    """

    Validator.validate_matrix_shape(matrix)
    Validator.validate_types(types)
    Validator.validate_input(matrix, types=types)

    try:
        nmatrix = np.zeros(matrix.shape)

        if 1 in types:
            nmatrix[:, types == 1] = matrix[:, types == 1] / \
                np.flip(np.sum(matrix[:, types == 1], axis=0))

        if -1 in types:
            nmatrix[:, types == -1] = (1/matrix[:, types == -1]) / \
                np.flip(np.sum(1/matrix[:, types == -1], axis=0))

        return nmatrix.astype(float)

    except Exception as e:
        raise RuntimeError(f"Failed to perform Sum normalization: {e}") from e

def max_normalization(
    matrix: np.ndarray | list,
    types: np.ndarray | list
) -> np.ndarray:
    """
    Normalize a Triangular Fuzzy Number decision matrix using maximum
    normalization.

    For benefit criteria, each fuzzy value is divided by the maximum value
    observed among alternatives. For cost criteria, the normalized value is
    obtained by applying a complement transformation.

    Parameters
    ----------
    matrix : np.ndarray | list
        TFN decision matrix with shape (m, n, 3).

    types : np.ndarray | list
        Criterion optimization directions:
        1 for benefit criteria,
        -1 for cost criteria.

    Returns
    -------
    np.ndarray
        Normalized TFN decision matrix with shape (m, n, 3).

    Raises
    ------
    ValueError
        If input matrix or criterion types are invalid.

    RuntimeError
        If normalization computation fails.
    """

    Validator.validate_matrix_shape(matrix)
    Validator.validate_types(types)
    Validator.validate_input(matrix, types=types)

    try:
        nmatrix = np.zeros(matrix.shape)

        if 1 in types:
            nmatrix[:, types == 1] = matrix[:, types == 1] / \
                np.max(matrix[:, types == 1], axis=0)

        if -1 in types:
            nmatrix[:, types == -1] = 1 - \
                (matrix[:, types == -1] / np.max(matrix[:, types == -1], axis=0))

        return nmatrix.astype(float)

    except Exception as e:
        raise RuntimeError(f"Failed to perform Max normalization: {e}") from e

def linear_normalization(
    matrix: np.ndarray | list,
    types: np.ndarray | list
) -> np.ndarray:
    """
    Normalize a Triangular Fuzzy Number decision matrix using linear
    normalization.

    Benefit criteria are normalized using the maximum value of each
    criterion, while cost criteria are normalized using the minimum value
    with reversed TFN components.

    Parameters
    ----------
    matrix : np.ndarray | list
        TFN decision matrix with shape (m, n, 3).

    types : np.ndarray | list
        Criterion optimization directions:
        1 for benefit criteria,
        -1 for cost criteria.

    Returns
    -------
    np.ndarray
        Normalized TFN decision matrix with shape (m, n, 3).

    Raises
    ------
    ValueError
        If input matrix or criterion types are invalid.

    RuntimeError
        If normalization computation fails.
    """

    Validator.validate_matrix_shape(matrix)
    Validator.validate_types(types)
    Validator.validate_input(matrix, types=types)
    
    try:
        nmatrix = np.zeros(matrix.shape)

        if 1 in types:
            nmatrix[:, types == 1] = matrix[:, types == 1] / \
                np.max(np.max(matrix[:, types == 1], axis=0), axis=1)[...,None]

        if -1 in types:
            nmatrix[:, types == -1] = np.min(np.min(matrix[:, types == -1], axis=0), axis=1)[...,None] / \
                matrix[:, types == -1][..., ::-1]

        return nmatrix.astype(float)

    except Exception as e:
        raise RuntimeError(f"Failed to perform Linear normalization: {e}") from e

def minmax_normalization(
    matrix: np.ndarray | list,
    types: np.ndarray | list
) -> np.ndarray:
    """
    Normalize a Triangular Fuzzy Number decision matrix using Min-Max
    normalization.

    The method transforms each criterion into a comparable scale using the
    minimum and maximum values observed in the decision matrix.

    For benefit criteria:

    .. math::

        r_{ij} =
        \\frac{x_{ij}-x_j^{min}}
        {x_j^{max}-x_j^{min}}

    For cost criteria, the transformation is reversed to preserve the
    preference direction.

    Parameters
    ----------
    matrix : np.ndarray | list
        TFN decision matrix with shape (m, n, 3).

    types : np.ndarray | list
        Criterion optimization directions:
        1 for benefit criteria,
        -1 for cost criteria.

    Returns
    -------
    np.ndarray
        Normalized TFN decision matrix with shape (m, n, 3).

    Raises
    ------
    ValueError
        If input matrix or criterion types are invalid.

    RuntimeError
        If normalization computation fails.
    """

    Validator.validate_matrix_shape(matrix)
    Validator.validate_types(types)
    Validator.validate_input(matrix, types=types)

    try:
        nmatrix = np.zeros(matrix.shape)
        if 1 in types:
            nmatrix[:, types == 1] = (matrix[:, types == 1] - np.min(matrix[:, types == 1, 0], axis=0)[..., None]) / \
                (np.max(matrix[:, types == 1, 2], axis=0) -
                    np.min(matrix[:, types == 1, 0], axis=0))[...,None]

        if -1 in types:
            nmatrix[:, types == -1] = ((matrix[:, types == -1] - np.max(matrix[:, types == -1, 2], axis=0)[...,None]) / (
                np.min(matrix[:, types == -1, 0], axis=0) - np.max(matrix[:, types == -1, 2], axis=0))[...,None])[..., ::-1]

        return nmatrix.astype(float)

    except Exception as e:
        raise RuntimeError(f"Failed to perform Min-Max normalization: {e}") from e

def vector_normalization(
    matrix: np.ndarray,
    *args
) -> np.ndarray:
    """
    Normalize a Triangular Fuzzy Number decision matrix using vector
    normalization.

    Each TFN component is normalized independently for every criterion:

    .. math::
        r_{ij}^{k} =
        \\frac{x_{ij}^{k}}
        {\\sqrt{\\sum_i (x_{ij}^{k})^2}}

    where ``k`` denotes the TFN component (lower, modal, upper).

    Parameters
    ----------
    matrix : np.ndarray, shape (m, n, 3)
        Decision matrix containing Triangular Fuzzy Numbers.

    *args :
        Additional arguments ignored by this normalization method.
        Included for compatibility with normalization interfaces requiring
        criterion types.

    Returns
    -------
    np.ndarray, shape (m, n, 3)
        Normalized TFN decision matrix.

    Raises
    ------
    ValueError
        If the input matrix cannot be converted into a numeric TFN matrix
        or does not have shape ``(m, n, 3)``.

    RuntimeError
        If normalization cannot be computed.
    """

    Validator.validate_matrix_shape(matrix)

    nmatrix = np.zeros(matrix.shape, dtype=object)
    try:
        for j in range(nmatrix.shape[1]):
            nmatrix[:, j] = matrix[:, j] / np.sqrt(np.sum(matrix[:, j]**2))
    
        return nmatrix.astype(float)

    except Exception as e:
        raise RuntimeError(f"Failed to compute Vector normalization: {e}") from e

def saw_normalization(
    matrix: np.ndarray,
    *args
) -> np.ndarray:
    """
    Normalize a Triangular Fuzzy Number decision matrix using
    Simple Additive Weighting (SAW) normalization.

    The normalization is performed independently for every criterion:

    .. math::
        r_{ij}^{k} =
        \\frac{x_{ij}^{k}}
        {\\max_i(x_{ij}^{k})}

    Parameters
    ----------
    matrix : np.ndarray, shape (m, n, 3)
        Decision matrix containing Triangular Fuzzy Numbers.

    *args :
        Additional arguments ignored by this normalization method.

    Returns
    -------
    np.ndarray, shape (m, n, 3)
        Normalized TFN decision matrix.

    Raises
    ------
    ValueError
        If the matrix is not a valid TFN decision matrix.

    RuntimeError
        If the normalization calculation fails.
    """

    Validator.validate_matrix_shape(matrix)

    try:
        nmatrix = np.zeros(matrix.shape, dtype=object)

        for j in range(nmatrix.shape[1]):
            nmatrix[:, j] = matrix[:, j] / np.max(matrix[:, j])
        
        return nmatrix.astype(float)

    except Exception as e:
        raise RuntimeError(f"Failed to compute SAW normalization: {e}") from e

def sqrt_normalization(
    matrix: np.ndarray,
    *args
) -> np.ndarray:
    """
    Normalize a Triangular Fuzzy Number decision matrix using square-root
    normalization.

    The normalization formula is:

    .. math::
        r_{ij}^{k} =
        \\frac{x_{ij}^{k}}
        {\\sqrt{\\frac{1}{3}\\sum_i(x_{ij}^{k})^2}}

    Parameters
    ----------
    matrix : np.ndarray, shape (m, n, 3)
        Decision matrix containing Triangular Fuzzy Numbers.

    *args :
        Additional arguments ignored by this normalization method.

    Returns
    -------
    np.ndarray, shape (m, n, 3)
        Normalized TFN decision matrix.

    Raises
    ------
    ValueError
        If the matrix cannot be converted into a numeric TFN matrix.

    RuntimeError
        If normalization cannot be performed.
    """

    Validator.validate_matrix_shape(matrix)

    try:
        nmatrix = np.zeros(matrix.shape, dtype=object)

        for j in range(nmatrix.shape[1]):
            nmatrix[:, j] = matrix[:, j] / np.sqrt(1/3 * np.sum(matrix[:, j]**2))

        return nmatrix.astype(float)

    except Exception as e:
        raise RuntimeError(f"Failed to compute Square-root normalization: {e}") from e

def cocoso_normalization(
    matrix: np.ndarray,
    types: np.ndarray | list
) -> np.ndarray:
    """
    Normalize a Triangular Fuzzy Number decision matrix using COCOSO
    normalization.

    For benefit criteria:

    .. math::
        r_{ij}^{k} =
        \\frac{x_{ij}^{k}-x_j^{min}}
        {x_j^{max}-x_j^{min}}

    For cost criteria:

    .. math::
        r_{ij}^{k} =
        \\frac{x_j^{max}-x_{ij}^{k}}
        {x_j^{max}-x_j^{min}}

    Parameters
    ----------
    matrix : np.ndarray, shape (m, n, 3)
        Decision matrix containing Triangular Fuzzy Numbers.

    types : np.ndarray | list, shape (n,)
        Criterion types:
        
        - ``1`` for benefit criteria,
        - ``-1`` for cost criteria.

    Returns
    -------
    np.ndarray, shape (m, n, 3)
        Normalized TFN decision matrix.

    Raises
    ------
    ValueError
        If input data have invalid dimensions or criterion types.

    RuntimeError
        If normalization cannot be computed.
    """

    Validator.validate_matrix_shape(matrix)
    Validator.validate_types(types)
    Validator.validate_input(matrix, types=types)

    try:
        nmatrix = np.zeros(matrix.shape, dtype=object)

        # profit criteria
        if 1 in types:
            nmatrix[:, types == 1] = (matrix[:, types == 1] - np.min(matrix[:, types == 1, 0], axis=0)[...,None]) / (np.max(matrix[:, types == 1, 2], axis=0) -
                    np.min(matrix[:, types == 1, 0], axis=0))[...,None]

        # cost criteria
        if -1 in types:
            nmatrix[:, types == -1] = (np.max(matrix[:, types == -1, 2], axis=0)[...,None] - matrix[:, types == -1][..., ::-1]) / (np.max(matrix[:, types == -1, 2], axis=0) -
                    np.min(matrix[:, types == -1, 0], axis=0))[...,None]

        return nmatrix.astype(float)

    except Exception as e:
        raise RuntimeError(f"Failed to compute COCOSO normalization: {e}") from e

def waspas_normalization(
    matrix: np.ndarray,
    types: np.ndarray | list
) -> np.ndarray:
    """
    Normalize a Triangular Fuzzy Number decision matrix using WASPAS
    normalization.

    For benefit criteria:

    .. math::
        r_{ij}^{k} =
        \\frac{x_{ij}^{k}}
        {\\max_i(x_{ij}^{u})}

    For cost criteria:

    .. math::
        r_{ij}^{k} =
        \\frac{\\min_i(x_{ij}^{l})}
        {x_{ij}^{k}}

    Parameters
    ----------
    matrix : np.ndarray, shape (m, n, 3)
        Decision matrix containing Triangular Fuzzy Numbers.

    types : np.ndarray | list, shape (n,)
        Criterion types:
        
        - ``1`` for benefit criteria,
        - ``-1`` for cost criteria.

    Returns
    -------
    np.ndarray, shape (m, n, 3)
        Normalized TFN decision matrix.

    Raises
    ------
    ValueError
        If the matrix or criterion types have invalid shape or values.

    RuntimeError
        If the normalization process fails.
    """

    Validator.validate_matrix_shape(matrix)
    Validator.validate_types(types)
    Validator.validate_input(matrix, types=types)

    try:
        nmatrix = np.zeros(matrix.shape, dtype=object)

        if 1 in types:
            nmatrix[:, types == 1] = matrix[:, types == 1] / np.max(matrix[:, types == 1, 2], axis=0)[...,None]

        if -1 in types:
            nmatrix[:, types == -1] = np.min(matrix[:, types == -1, 0], axis=0)[...,None] / matrix[:, types == -1]
            
        return nmatrix.astype(float)

    except Exception as e:
        raise RuntimeError(f"Failed to compute WASPAS normalization: {e}") from e