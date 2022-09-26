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

            a : float
                Threshold parameter

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
    wmatrix = nmatrix[:, :, :] * weights

    # aggregated profit and cost values
    Tp = np.sum(wmatrix[:, types == 1], axis=1)
    Tm = np.sum(wmatrix[:, types == -1], axis=1)

    # distance
    Q = Tp + (np.sum(Tm)) / (Tm * np.sum(np.divide(1,Tm.astype(float))))
    if np.isnan(np.max(Q.astype(float).ravel())):
        Q = np.nan_to_num(Q.ravel().astype(float)).reshape(Tp.shape)

    # defuzzified values
    Q = Q[:, 0] + ((Q[:, 2] - Q[:, 0]) - (Q[:, 1] - Q[:, 2])) / 3
    return Q / np.max(Q) 