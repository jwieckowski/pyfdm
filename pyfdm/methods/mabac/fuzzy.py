# Copyright (c) 2022 Jakub Więckowski

import numpy as np

def fuzzy(matrix, weights, types, normalization, defuzzify):
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

            defuzzify: callable
                Function used to defuzzify the TFN into crisp value

        Returns
        -------
            ndarray:
                Crisp preferences of alternatives

    """

    # normalized decision matrix
    nmatrix = normalization(matrix, types)
    
    if weights.ndim == 1:
        weights = np.repeat(weights, 3).reshape((len(weights), 3))

    # weighted normalized decision matrix
    wmatrix = nmatrix * weights + weights

    # approximate border area matrix
    G = np.product(wmatrix, axis=0) ** (1/wmatrix.shape[0])

    # distance
    Q = wmatrix - G[..., ::-1]
    
    # preference value
    S = np.array([np.sum(q, axis=0) for q in Q])
    return np.array([defuzzify(s) for s in S])
