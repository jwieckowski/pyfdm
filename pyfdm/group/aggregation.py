# Copyright (c) 2026 Jakub Więckowski

import numpy as np

__all__ = ['aggregate', 'geometric_mean', 'arithmetic_mean', 'weighted_average', 'owa']

def aggregate(
    matrices: list, 
    method: str = 'geometric',
    weights=None, 
    owa_weights=None
) -> np.ndarray:
    """
    Aggregate a list of TFN decision matrices from multiple experts.

    Delegates per-cell aggregation to pyfdm.TFN.aggregation operators,
    ensuring consistency between group-level and TFN-level aggregation.

    Parameters
    ----------
        matrices : list of ndarray, each shape (m, n, 3)
            TFN decision matrices from E experts.

        method : str, default 'geometric'
            - 'geometric'  — geometric mean (recommended for F-AHP)
            - 'arithmetic' — arithmetic mean
            - 'weighted'   — weighted arithmetic mean (*weights* required)
            - 'owa'        — Ordered Weighted Average (*owa_weights* required)

        weights : list or ndarray, shape (E,), optional
            Expert importance weights for method='weighted'.
            Normalized to sum to 1 automatically.

        owa_weights : list or ndarray, shape (E,), optional
            OWA operator weights for method='owa'.

    Returns
    -------
        ndarray, shape (m, n, 3)
            Aggregated TFN decision matrix.

    Raises
    ------
        ValueError - on invalid method or missing required arguments.

    Examples
    --------
        >>> from pyfdm.group import aggregate
        >>> agg = aggregate(matrices, method='geometric')
        >>> agg = aggregate(matrices, method='weighted', weights=[0.5, 0.3, 0.2])
    """
    if not matrices:
        raise ValueError('matrices list is empty.')

    stack = np.stack(matrices, axis=0)   # (E, m, n, 3)

    if method == 'geometric':
        return geometric_mean(stack)
    elif method == 'arithmetic':
        return arithmetic_mean(stack)
    elif method == 'weighted':
        if weights is None:
            raise ValueError("method='weighted' requires expert weights.")
        return weighted_average(stack, weights)
    elif method == 'owa':
        if owa_weights is None:
            raise ValueError("method='owa' requires owa_weights.")
        return owa(stack, owa_weights)
    else:
        raise ValueError(
            f"Unknown aggregation method '{method}'. "
            "Choose from: 'geometric', 'arithmetic', 'weighted', 'owa'."
        )

def geometric_mean(stack: np.ndarray) -> np.ndarray:
    """
    Geometric mean aggregation of TFN matrices: GM(x₁,...,xₑ) = (∏ xₑ)^(1/E).

    Applies the geometric mean independently to each (l, m, u) component
    across all experts, for every cell (alternative, criterion) in the matrix.

    Uses the same log-sum-exp approach as pyfdm.TFN.aggregation.tfn_wg
    with equal weights.

    Parameters
    ----------
        stack : ndarray, shape (E, m, n, 3)

    Returns
    -------
        ndarray, shape (m, n, 3)

    Notes
    -----
        Recommended for F-AHP pairwise comparison aggregation (Buckley, 1985).

    Examples
    --------
        >>> import numpy as np
        >>> from pyfdm.group import geometric_mean
        >>> # 2 experts, 1 alternative, 1 criterion, TFN (l, m, u)
        >>> stack = np.array([
        ...     [[[1, 2, 3]]],
        ...     [[[3, 4, 5]]],
        ... ])
        >>> geometric_mean(stack)
        array([[[1.73205081, 2.82842712, 3.87298335]]])
    """
    eps = 1e-10
    safe = np.where(stack <= 0, eps, stack)
    return np.exp(np.mean(np.log(safe), axis=0))


def arithmetic_mean(stack: np.ndarray) -> np.ndarray:
    """
    Arithmetic mean aggregation of TFN matrices: AM = (1/E) Σ xₑ.

    Parameters
    ----------
        stack : ndarray, shape (E, m, n, 3)

    Returns
    -------
        ndarray, shape (m, n, 3)

    Examples
    --------
        >>> import numpy as np
        >>> from pyfdm.group import arithmetic_mean
        >>> # 2 experts, 1 alternative, 1 criterion, TFN (l, m, u)
        >>> stack = np.array([
        ...     [[[1, 2, 3]]],
        ...     [[[3, 4, 5]]],
        ... ])
        >>> arithmetic_mean(stack)
        array([[[2., 3., 4.]]])
    """
    return np.mean(stack, axis=0)


def weighted_average(stack: np.ndarray, weights) -> np.ndarray:
    """
    Weighted arithmetic mean aggregation of TFN matrices.

    Equivalent to applying pyfdm.TFN.aggregation.tfn_wa (with the given
    weights) independently to each cell of the matrix.

    Parameters
    ----------
        stack : ndarray, shape (E, m, n, 3)
        weights : array-like, shape (E,)
            Expert importance weights (normalized to sum to 1 automatically).

    Returns
    -------
        ndarray, shape (m, n, 3)

    Examples
    --------
        >>> import numpy as np
        >>> from pyfdm.group import weighted_average
        >>> # 2 experts, 1 alternative, 1 criterion, TFN (l, m, u)
        >>> stack = np.array([
        ...     [[[1, 2, 3]]],
        ...     [[[3, 4, 5]]],
        ... ])
        >>> weighted_average(stack, weights=[0.75, 0.25])
        array([[[1.5, 2.5, 3.5]]])
    """
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != stack.shape[0]:
        raise ValueError(
            f'Length of weights ({w.shape[0]}) must match '
            f'number of experts ({stack.shape[0]}).'
        )
    if np.any(w < 0):
        raise ValueError('Expert weights must be non-negative.')
    w = w / w.sum()
    return np.einsum('e,emnk->mnk', w, stack)


def owa(stack: np.ndarray, owa_weights) -> np.ndarray:
    """
    Ordered Weighted Average (OWA) aggregation of TFN matrices.

    For each cell (i, j) and each TFN component (l, m, u), expert values
    are sorted in descending order and OWA weights are applied.

    Equivalent to applying pyfdm.TFN.aggregation.tfn_owa (with the given
    weights) independently to each cell.

    Parameters
    ----------
        stack : ndarray, shape (E, m, n, 3)
        owa_weights : array-like, shape (E,)
            OWA operator weights (applied to descending-sorted values).

    Returns
    -------
        ndarray, shape (m, n, 3)

    Examples
    --------
        >>> import numpy as np
        >>> from pyfdm.group import owa
        >>> # 3 experts, 1 alternative, 1 criterion, TFN (l, m, u)
        >>> stack = np.array([
        ...     [[[1, 1, 1]]],
        ...     [[[3, 3, 3]]],
        ...     [[[5, 5, 5]]],
        ... ])
        >>> # weights favor the largest values (descending order)
        >>> owa(stack, owa_weights=[0.5, 0.3, 0.2])
        array([[[3.6, 3.6, 3.6]]])
    """
    w = np.asarray(owa_weights, dtype=float)
    if w.shape[0] != stack.shape[0]:
        raise ValueError(
            f'Length of owa_weights ({w.shape[0]}) must match '
            f'number of experts ({stack.shape[0]}).'
        )
    if np.any(w < 0):
        raise ValueError('OWA weights must be non-negative.')
    w = w / w.sum()

    sorted_stack = np.sort(stack, axis=0)[::-1, ...]    # (E, m, n, 3)
    return np.einsum('e,emnk->mnk', w, sorted_stack)