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

    # normalized decision matrix
    nmatrix = normalization(matrix, types)

    if weights.ndim == 1:
        weights = np.repeat(weights, 3).reshape((len(weights), 3))

    # weighted normalized decision matrix
    wmatrix = nmatrix * weights

    # profit and cost overall ratings
    Sp = np.sum(wmatrix[:, types == 1], axis=1)
    Sm = np.sum(wmatrix[:, types == -1], axis=1)

    # preference value
    S = np.array([np.sqrt(1/3 * ((sp[0]-sm[0])**2 + (sp[1]-sm[1])
                                 ** 2 + (sp[2]-sm[2])**2)) for sp, sm in zip(Sp, Sm)])
    return S
