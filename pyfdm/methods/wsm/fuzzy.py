# Copyright (c) 2023 Jakub Więckowski

import numpy as np

def fuzzy(matrix, weights, normalization, defuzzify):
    """
        Calculates the alternatives preferences based on Triangular Fuzzy Number extension

        Parameters
        ----------
            matrix : ndarray
                Decision matrix / alternatives data.
                Alternatives are in rows and Criteria are in columns.

            weights : ndarray
                Vector of criteria weights in a crisp form

            normalization: callable
                Function used to normalize the decision matrix
                
        Returns
        -------
            ndarray:
                Crisp preferences of alternatives

    """

    # normalized decision matrix
    if normalization is not None:
        nmatrix = normalization(matrix)
    else:
        nmatrix = matrix.copy()

    if weights.ndim == 1:
        weights = np.repeat(weights, 3).reshape((len(weights), 3))

    # weighted normalized decision matrix
    wmatrix = nmatrix * weights

    sum_w = np.sum(wmatrix, axis=1)

    return np.array([defuzzify(s) for s in sum_w])

