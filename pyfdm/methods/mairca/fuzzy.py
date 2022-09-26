# Copyright (c) 2022 Jakub Więckowski

import numpy as np

def fuzzy(matrix, weights, types, normalization, distance):
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

            distance: callable
                Function used to calculate distance between two Triangular Fuzzy Numbers

        Returns
        -------
            ndarray:
                Crisp preferences of alternatives

    """

    # alternative selection propability
    P = 1 / matrix.shape[0]

    # Fuzzy theoretical evaluation matrix
    tpa = np.ones((matrix.shape), dtype=object)
    for j in range(matrix.shape[1]):
        tpa[:, j] = P * weights[j]

    # normalized fuzzy decision matrix
    nmatrix = normalization(matrix, types)

    # fuzzy elements of the actual ponder matrix
    tra = nmatrix * tpa

    # distance between Fuzzy Numbers
    d = np.zeros((matrix.shape[0], matrix.shape[1], 1))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            d[i, j] = distance(tpa[i, j], tra[i, j])

    # preference value
    Q = np.sum(d, axis=1).flatten()
    return Q
