# Copyright (c) 2022 Jakub Więckowski

import numpy as np

def fuzzy(matrix, weights, types, normalization):
    """
        Calculates the alternatives preferences based on Triangular Fuzzy Number extension

        Parameters
        ----------
            matrix : ndarray
                Decision matrix / alternatives data.
                Alternatives are in rows and Criteria are in columns.

            weights : ndarray
                Vector of criteria weights in a crisp form

            types : ndarray
                Types of criteria, 1 profit, -1 cost

            normalization: callable
                Function used to normalize the decision matrix
                
        Returns
        -------
            ndarray:
                Crisp preferences of alternatives

    """

    # extended decision matrix
    exmatrix = np.zeros(
        (matrix.shape[0]+1, matrix.shape[1], matrix.shape[2]), dtype=object)
    exmatrix[1:] = matrix

    exmatrix[0, :] = np.repeat(np.max(np.max(matrix, axis=0), axis=1), 3).reshape(
        matrix.shape[1], matrix.shape[2])

    # normalized decision matrix
    nmatrix = normalization(exmatrix, types)

    if weights.ndim == 1:
        weights = np.repeat(weights, 3).reshape((len(weights), 3))

    # weighted normalized decision matrix
    wmatrix = nmatrix * weights

    # overall preference
    S = 1/3 * np.sum(np.sum(wmatrix, axis=1), axis=1)
    return S[1:] / S[0]
