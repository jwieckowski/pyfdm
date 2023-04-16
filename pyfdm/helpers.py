# Copyright (c) 2022 Jakub Więckowski

import numpy as np

__all__ = [
    'rank',
    'generate_fuzzy_matrix'
]


def rank(x, descending=True):
    """
        Calculates ranking of given values with the given direction, default descending order

        Parameters
        ----------
            x: ndarray
                Array with values

            descending: boolean, default=True
                Switch to change ranking order

        Returns
        -------
            ndarray
                Ranking with given order

    """
    try:
        s = [sorted(x, reverse=descending).index(r)+1 for r in x]
    except:
        raise ValueError('Error occurred in ranking calculation')
    return np.array([(ss * s.count(ss) + s.count(ss) - 1) / s.count(ss) if s.count(ss) <= 2 else np.sum(list(range(ss, ss+s.count(ss)))) / s.count(ss) for ss in s])


def generate_fuzzy_matrix(m, n, lower=0.0, upper=1.0):
    """
        Generates random Triangular Fuzzy Numbers with m alternatives and n criteria, each TFN is places between lower and upper bound

        Parameters
        ----------
            m: int
                Number of alternatives

            n: int
                Number of criteria

            lower: float, default=0.0
                Minimum value of left bound

            upper: float, default=1.0
                Maximum value of right bound

        Returns
        -------
            ndarray
                Matrix with random TFN within given bounds

    """

    if lower > upper:
        raise ValueError("Lower bound of TFN must be greater than upper")

    matrix = np.random.uniform(low=lower, high=upper, size=(m, n, 3))
    return np.array([np.sort(m, axis=1) for m in matrix])
