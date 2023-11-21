# Copyright (c) 2023 Jakub Więckowski

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
    wsm_wmatrix = nmatrix * weights
    wpm_wmatrix = nmatrix ** weights[..., ::-1]

    # calculation of optimality function values
    Q = np.sum(wsm_wmatrix, axis=1)
    P = np.prod(wpm_wmatrix, axis=1)

    # deffuzify values
    Q_def = np.array([defuzzify(q) for q in Q])
    P_def = np.array([defuzzify(p) for p in P])

    d = np.sum(P_def) / (np.sum(Q_def) + np.sum(P_def))

    # value of integrated utility
    K = d * Q_def + (1-d) * P_def

    return K
