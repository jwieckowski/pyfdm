# Copyright (c) 2022-2023 Jakub Więckowski

import numpy as np

__all__ = [
    'equal_weights',
    'shannon_entropy_weights',
    'standard_deviation_weights',
    'variance_weights'
]


def equal_weights(matrix):
    """
        Calculates the objective weights for Triangular Fuzzy Matrix, each weight will have the same value

        Parameters
        ----------
            matrix: ndarray
                Decision matrix / alternatives data
                Alternatives are in rows and Criteria are in columns

        Returns
        -------
            ndarray
                Array of equal weights
    """

    w = np.ones(matrix.shape[1]) / matrix.shape[1]
    return np.repeat(w, 3).reshape((len(w), 3))


def shannon_entropy_weights(matrix):
    """
        Calculates the objective weights for Triangular Fuzzy Matrix, weight depend on the entropy measure in the column

        Parameters
        ----------
            matrix: ndarray
                Decision matrix / alternatives data
                Alternatives are in rows and Criteria are in columns

        Returns
        -------
            ndarray
                Array of weights based on matrix entropy
    """
    # https://www.mdpi.com/1099-4300/19/7/373/htm

    # shannon entropy vector
    e = np.zeros((matrix.shape[1], 3))
    for j in range(matrix.shape[1]):
        e[j] = - 1/(np.log(matrix.shape[0])) * \
            np.sum(matrix[:, j] * np.log(matrix[:, j]), axis=0)

    # fuzzy diversification vector
    d = 1 - e

    # fuzzy criteria weights
    w = d / np.sum(d, axis=0)

    return w


def standard_deviation_weights(matrix):
    """
        Calculates the objective weights for Triangular Fuzzy Matrix, weight depend on the data standard deviation in the column

        Parameters
        ----------
            matrix: ndarray
                Decision matrix / alternatives data
                Alternatives are in rows and Criteria are in columns

        Returns
        -------
            ndarray
                Array of weights based on matrix entropy
    """
    w = np.std(matrix, axis=0)
    return w / np.sum(w, axis=0)


def variance_weights(matrix):
    """
        Calculates the objective weights for Triangular Fuzzy Matrix, weight depend on the data variance in the column

        Parameters
        ----------
            matrix: ndarray
                Decision matrix / alternatives data
                Alternatives are in rows and Criteria are in columns

        Returns
        -------
            ndarray
                Array of weights based on matrix entropy
    """
    w = np.var(matrix, axis=0)
    return w / np.sum(w, axis=0)
