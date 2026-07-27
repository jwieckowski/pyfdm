# Copyright (c) 2022 - 2026 Jakub Więckowski

import numpy as np

__all__ = ['rank_alternatives']

def rank_alternatives(
    scores, 
    descending: bool = True,
    method: str = 'average'
) -> np.ndarray:
    """
    Compute the ranking of alternatives based on a vector of scores.

    Unlike a method-specific ``.rank()`` (which requires first running
    an MCDA method), this function works directly on any 1-D array of
    crisp preference values — e.g. expert ratings, or externally computed
    utility scores.

    Parameters
    ----------
        scores : array-like, shape (m,)
            Crisp preference/utility score for each alternative.

        descending : bool, default True
            If True, the highest score receives rank 1 (best).
            If False, the lowest score receives rank 1 (e.g. for
            distance- or cost-based scores where lower is better).

        method : str, default 'average'
            How to handle tied scores:
            - 'average'  : tied alternatives share the average of their ranks
                           (e.g. two 2nd-place ties both get rank 2.5)
            - 'min'      : tied alternatives all get the best (lowest) rank
                           in the tie group (competition ranking, "1224")
            - 'max'      : tied alternatives all get the worst (highest) rank
                           in the tie group ("1334")
            - 'dense'    : ranks are consecutive integers with no gaps for
                           ties ("1223")
            - 'ordinal'  : ties are broken by original array order, every
                           alternative gets a unique rank ("1234")

    Returns
    -------
        ndarray, shape (m,)
            Ranking position for each alternative (1 = best, per
            *descending*). Dtype is float for 'average', int otherwise.

    Raises
    ------
        TypeError  - if scores cannot be converted to a numeric ndarray.
        ValueError - if scores is empty, not 1-D, contains NaN/Inf,
                     or *method* is not recognized.

    Examples
    --------
        >>> from pyfdm.utils import rank_alternatives
        >>> scores = [0.82, 0.55, 0.91, 0.55]
        >>> rank_alternatives(scores)
        array([2. , 3.5, 1. , 3.5])

        >>> # Distance-based scores: lower is better
        >>> rank_alternatives([0.1, 0.4, 0.05], descending=False)
        array([2., 3., 1.])

        >>> # Competition-style ranking for ties
        >>> rank_alternatives([10, 10, 8], method='min')
        array([1, 1, 3])
    """
    def _validate_scores(scores) -> np.ndarray:
        try:
            arr = np.asarray(scores, dtype=float)
        except (TypeError, ValueError) as e:
            raise TypeError(f'scores must be convertible to a numeric array, got {type(scores).__name__}.') from e

        if arr.ndim != 1:
            raise ValueError(f'scores must be a 1-D array of per-alternative values, got shape {arr.shape}.')
        if arr.shape[0] == 0:
            raise ValueError('scores must contain at least one alternative.')
        if np.any(np.isnan(arr)):
            raise ValueError('scores contains NaN values; ranking is undefined.')
        if np.any(np.isinf(arr)):
            raise ValueError('scores contains Inf values; ranking is undefined.')
        return arr

    def _validate_method(method: str) -> None:
        valid = ('average', 'min', 'max', 'dense', 'ordinal')
        if method not in valid:
            raise ValueError(f"Unknown tie-breaking method '{method}'. Choose from: {valid}.")

    arr = _validate_scores(scores)
    _validate_method(method)

    n = arr.shape[0]

    # Sort order: best-first according to `descending`
    order = np.argsort(-arr if descending else arr, kind='stable')

    if method == 'ordinal':
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)
        return ranks

    # Base ranks via ordinal position, then adjust for ties.
    sorted_scores = arr[order]

    # Identify tie groups in sorted order
    ranks_sorted = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        # extend the group while values are equal
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        positions = np.arange(i + 1, j + 2)   # 1-based ordinal positions in this group

        if method == 'average':
            ranks_sorted[i:j + 1] = positions.mean()
        elif method == 'min':
            ranks_sorted[i:j + 1] = positions.min()
        elif method == 'max':
            ranks_sorted[i:j + 1] = positions.max()
        elif method == 'dense':
            # filled in a second pass below
            ranks_sorted[i:j + 1] = -1  # placeholder, replaced below
        i = j + 1

    if method == 'dense':
        # Dense rank: consecutive integers per distinct value
        dense_ranks_sorted = np.empty(n, dtype=int)
        current_rank = 0
        prev_val = None
        for idx in range(n):
            val = sorted_scores[idx]
            if prev_val is None or val != prev_val:
                current_rank += 1
                prev_val = val
            dense_ranks_sorted[idx] = current_rank
        ranks_sorted = dense_ranks_sorted

    # Scatter back to original order
    ranks = np.empty(n, dtype=ranks_sorted.dtype)
    ranks[order] = ranks_sorted

    if method in ('min', 'max', 'dense'):
        return ranks.astype(int)
    return ranks.astype(float)
