# Copyright (c) 2022 Jakub Więckowski

import numpy as np

def fuzzy(matrix, weights, types, defuzzify):
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

            defuzzify: callable
                Function used to defuzzify the TFN into crisp value

        Returns
        -------
            ndarray:
                Crisp preferences of alternatives

    """

    # cost fuzzy performance rating
    Is = np.zeros((matrix.shape[0], matrix.shape[2]))
    for i in range(matrix.shape[0]):
        Is[i] = np.sum([weights[j] * ((np.max(matrix[:, j], axis=0) - matrix[i, j][..., ::-1]) /
                       (np.min(matrix[:, j], axis=0))) for j in range(matrix.shape[1]) if types[j] == -1], axis=0)

    # cost fuzzy linear performance rating
    Iss = Is - np.min(Is, axis=0)[..., ::-1]

    # profit fuzzy performance rating
    Os = np.zeros((matrix.shape[0], matrix.shape[2]))
    for i in range(matrix.shape[0]):
        Os[i] = np.sum([weights[j] * ((matrix[i, j] - np.min(matrix[:, j][..., ::-1], axis=0)) /
                       (np.min(matrix[:, j], axis=0))) for j in range(matrix.shape[1]) if types[j] == 1], axis=0)

    # profit fuzzy linear performance rating
    Oss = Os - np.min(Os, axis=0)[..., ::-1]
    
    # aggregate fuzzy performance rating
    P = Iss + Oss - np.min(Iss + Oss, axis=0)[..., ::-1]

    return np.array([defuzzify(p) for p in P])
