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
                Function used to calculate distance from fuzzy negative solution

        Returns
        -------
            ndarray:
                Crisp preferences of alternatives

    """

    # Normalized fuzzy decision matrix
    nmatrix = normalization(matrix, types)

    if weights.ndim == 1:
        weights = np.repeat(weights, 3).reshape((len(weights), 3))

    # Weighted normalized fuzzy decision matrix
    wmatrix = nmatrix * weights

    # Fuzzy positive ideal solution and fuzzy negative ideal solution
    ideal = np.ones((matrix.shape[1], matrix.shape[2]))
    nideal = np.zeros((matrix.shape[1], matrix.shape[2]))

    # Distance to FPIS and FNIS
    fpis, fnis = np.zeros((matrix.shape[0], ), dtype=object), np.zeros(
        (matrix.shape[0], ), dtype=object)
    for i in range(matrix.shape[0]):
        fpis[i] = np.sum([distance(wmatrix[i, j], ideal[j])
                          for j in range(matrix.shape[1])])
        fnis[i] = np.sum([distance(wmatrix[i, j], nideal[j])
                          for j in range(matrix.shape[1])])

    return fnis / (fpis + fnis)
