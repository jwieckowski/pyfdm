# Copyright (c) 2022 Jakub Więckowski

import numpy as np

__all__ = [
    'spearman_coef',
    'pearson_coef',
    'weighted_spearman_coef',
    'ws_rank_similarity_coef'
]


def spearman_coef(x, y):
    """
        Calculate Spearman correlation between two vectors

        Parameters
        ----------
            x : ndarray
                Array with values

            y : ndarray
                Array with values

        Returns
        -------
            float
                Correlation between two vectors
    """
    return (np.cov(x, y, bias=True)[0][1]) / (np.std(x) * np.std(y))


def pearson_coef(x, y):
    """
        Calculate Pearson correlation between two vectors

        Parameters
        ----------
            x : ndarray
                Array with values

            y : ndarray
                Array with values

        Returns
        -------
            float
                Correlation between two vectors
    """
    return (np.cov(x, y, bias=True)[0][1]) / (np.std(x) * np.std(y))


def weighted_spearman_coef(x, y):
    """
        Calculate Weighted Spearman correlation between two rankings

        Parameters
        ----------
            x : ndarray
                Array with ranking

            y : ndarray
                Array with ranking

        Returns
        -------
            float
                Correlation between two vectors
    """
    N = x.shape[0]
    return 1 - ((6 * np.sum((x-y)**2 * ((N - x + 1) + (N - y + 1)))) / (N**4 + N**3 - N**2 - N))


def ws_rank_similarity_coef(x, y):
    """
        Calculate WS Rank Similarity Coefficient between two rankings

        Parameters
        ----------
            x : ndarray
                Array with ranking

            y : ndarray
                Array with ranking

        Returns
        -------
            float
                Correlation between two rankings
    """
    N = x.shape[0]
    return 1 - np.sum(2.0**(-1.0 * x) * (np.fabs(x - y)) / (np.max((np.fabs(1 - x), np.fabs(N - x)), axis=0)))
