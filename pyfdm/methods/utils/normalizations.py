# Copyright (c) 2022-2023 Jakub Więckowski

import numpy as np

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


def sum_normalization(matrix, types):
    """
        Calculates the normalized value of Triangular Fuzzy matrix using sum normalization

        Parameters
        ----------
            matrix : ndarray
                Matrix with Triangular Fuzzy Numbers

            types : ndarray
                Types of criteria, 1 profit, -1 cost

        Returns
        -------
            ndarray
                Normalized Triangular Fuzzy matrix
    """
    nmatrix = np.zeros(matrix.shape, dtype=object)

    # profit criteria
    if 1 in types:
        nmatrix[:, types == 1] = matrix[:, types == 1] / \
            np.flip(np.sum(matrix[:, types == 1], axis=0))

    # cost criteria
    if -1 in types:
        nmatrix[:, types == -1] = (1/matrix[:, types == -1]) / \
            np.flip(np.sum(1/matrix[:, types == -1], axis=0))

    return nmatrix.astype(float)


def max_normalization(matrix, types):
    """
        Calculates the normalized value of Triangular Fuzzy matrix using max normalization

        Parameters
        ----------
            matrix : ndarray
                Matrix with Triangular Fuzzy Numbers

            types : ndarray
                Types of criteria, 1 profit, -1 cost

        Returns
        -------
            ndarray
                Normalized Triangular Fuzzy matrix
    """
    nmatrix = np.zeros(matrix.shape, dtype=object)

    # profit criteria
    if 1 in types:
        nmatrix[:, types == 1] = matrix[:, types == 1] / \
            np.max(matrix[:, types == 1], axis=0)

    # cost criteria
    if -1 in types:
        nmatrix[:, types == -1] = 1 - \
            (matrix[:, types == -1] / np.max(matrix[:, types == -1], axis=0))

    return nmatrix.astype(float)


def linear_normalization(matrix, types):
    """
        Calculates the normalized value of Triangular Fuzzy matrix using linear normalization

        Parameters
        ----------
            matrix : ndarray
                Matrix with Triangular Fuzzy Numbers

            types : ndarray
                Types of criteria, 1 profit, -1 cost

        Returns
        -------
            ndarray
                Normalized Triangular Fuzzy matrix
    """
    nmatrix = np.zeros(matrix.shape, dtype=object)

    # profit criteria
    if 1 in types:
        nmatrix[:, types == 1] = matrix[:, types == 1] / \
            np.max(np.max(matrix[:, types == 1], axis=0), axis=1)[...,None]

    # cost criteria
    if -1 in types:
        nmatrix[:, types == -1] = np.min(np.min(matrix[:, types == -1], axis=0), axis=1)[...,None] / \
            matrix[:, types == -1][..., ::-1]

    return nmatrix.astype(float)


def minmax_normalization(matrix, types):
    """
        Calculates the normalized value of Triangular Fuzzy matrix using Min-Max normalization

        Parameters
        ----------
            matrix : ndarray
                Matrix with Triangular Fuzzy Numbers

            types : ndarray
                Types of criteria, 1 profit, -1 cost

        Returns
        -------
            ndarray
                Normalized Triangular Fuzzy matrix
    """
    nmatrix = np.zeros(matrix.shape, dtype=object)

    # profit criteria
    if 1 in types:
        nmatrix[:, types == 1] = (matrix[:, types == 1] - np.min(matrix[:, types == 1, 0], axis=0)[..., None]) / \
            (np.max(matrix[:, types == 1, 2], axis=0) -
                np.min(matrix[:, types == 1, 0], axis=0))[...,None]

    # cost criteria
    if -1 in types:
        nmatrix[:, types == -1] = ((matrix[:, types == -1] - np.max(matrix[:, types == -1, 2], axis=0)[...,None]) / (
            np.min(matrix[:, types == -1, 0], axis=0) - np.max(matrix[:, types == -1, 2], axis=0))[...,None])[..., ::-1]

    return nmatrix.astype(float)

def vector_normalization(matrix, *args):
    """
        Calculates the normalized value of Triangular Fuzzy matrix using vector normalization

        Parameters
        ----------
            matrix : ndarray
                Matrix with Triangular Fuzzy Numbers

            *args is necessary for methods which reqiure some additional data
        Returns
        -------
            ndarray
                Normalized Triangular Fuzzy matrix
    """
    nmatrix = np.zeros(matrix.shape, dtype=object)

    # for each column
    for j in range(nmatrix.shape[1]):
        nmatrix[:, j] = matrix[:, j] / np.sqrt(np.sum(matrix[:, j]**2))

    return nmatrix.astype(float)

def saw_normalization(matrix, *args):
    """
        Calculates the normalized value of Triangular Fuzzy matrix using simple addictive weight normalization

        Parameters
        ----------
            matrix : ndarray
                Matrix with Triangular Fuzzy Numbers

            *args is necessary for methods which reqiure some additional data
        Returns
        -------
            ndarray
                Normalized Triangular Fuzzy matrix
    """
    nmatrix = np.zeros(matrix.shape, dtype=object)

    # for each column
    for j in range(nmatrix.shape[1]):
        nmatrix[:, j] = matrix[:, j] / np.max(matrix[:, j])

    return nmatrix.astype(float)

def sqrt_normalization(matrix, *args):
    """
        Calculates the normalized value of Triangular Fuzzy matrix using sqrt normalization

        Parameters
        ----------
            matrix : ndarray
                Matrix with Triangular Fuzzy Numbers

            *args is necessary for methods which reqiure some additional data
        Returns
        -------
            ndarray
                Normalized Triangular Fuzzy matrix
    """
    nmatrix = np.zeros(matrix.shape, dtype=object)

    # for each column
    for j in range(nmatrix.shape[1]):
        nmatrix[:, j] = matrix[:, j] / np.sqrt(1/3 * np.sum(matrix[:, j]**2))

    return nmatrix.astype(float)

def waspas_normalization(matrix, types):
    """
        Calculates the normalized value of Triangular Fuzzy matrix using WASPAS normalization

        Parameters
        ----------
            matrix : ndarray
                Matrix with Triangular Fuzzy Numbers

            types : ndarray
                Types of criteria, 1 profit, -1 cost

        Returns
        -------
            ndarray
                Normalized Triangular Fuzzy matrix
    """
    nmatrix = np.zeros(matrix.shape, dtype=object)

    # profit criteria
    if 1 in types:
        nmatrix[:, types == 1] = matrix[:, types == 1] / np.max(matrix[:, types == 1, 2], axis=0)[...,None]

    # cost criteria
    if -1 in types:
        nmatrix[:, types == -1] = np.min(matrix[:, types == -1, 0], axis=0)[...,None] / matrix[:, types == -1]
        
    return nmatrix.astype(float)

def cocoso_normalization(matrix, types):
    """
        Calculates the normalized value of Triangular Fuzzy matrix using COCOSO normalization

        Parameters
        ----------
            matrix : ndarray
                Matrix with Triangular Fuzzy Numbers

            types : ndarray
                Types of criteria, 1 profit, -1 cost

        Returns
        -------
            ndarray
                Normalized Triangular Fuzzy matrix
    """
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
