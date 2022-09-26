# Copyright (c) 2022 Jakub Więckowski

import numpy as np

def fuzzy(matrix, weights, types, defuzzify, v):
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

            v : float
                Weight of the strategy (see VIKOR algorithm explanation).

        Returns
        -------
            ndarray:
                Crisp preferences of alternatives

    """

    # ideal and nadir values
    ideal, nadir = np.zeros((matrix.shape[1], 3)), np.zeros(
        (matrix.shape[1], 3))
    for j in range(matrix.shape[1]):
        if types[j] == 1:
            ideal[j] = np.max(matrix[:, j], axis=0)
            nadir[j] = np.min(matrix[:, j], axis=0)
        else:
            ideal[j] = np.min(matrix[:, j], axis=0)
            nadir[j] = np.max(matrix[:, j], axis=0)

    # normalized fuzzy difference
    d = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if types[j] == 1:
                d[i, j] = (ideal[j] - np.flipud(matrix[i, j])) / \
                    (ideal[j, 2] - nadir[j, 0])
            else:
                d[i, j] = (matrix[i, j] - np.flipud(ideal[j])) / \
                    (nadir[j, 2] - ideal[j, 0])

    if weights.ndim == 1:
        weights = np.repeat(weights, 3).reshape((len(weights), 3))

    # S, R, Q rankings
    S, R, Q = np.zeros((matrix.shape[0], 3)), np.zeros(
        (matrix.shape[0], 3)), np.zeros((matrix.shape[0], 3))

    for i in range(matrix.shape[0]):
        S[i] = np.sum(d[i, :] * weights, axis=0)
        R[i] = np.max(d[i, :] * weights, axis=0)

    for i in range(matrix.shape[0]):
        Q[i] = v * (S[i] - np.flipud(np.min(S, axis=0)))/(np.max(S, axis=0)[2] - np.min(S, axis=0)[0]) + \
            (1-v)*(R[i] - np.flipud(np.min(R, axis=0))) / \
            (np.max(R, axis=0)[2] - np.min(R, axis=0)[0])

    # defuzzification
    crisp_S = np.array([defuzzify(s) for s in S])
    crisp_R = np.array([defuzzify(r) for r in R])
    crisp_Q = np.array([defuzzify(q) for q in Q])

    return crisp_S, crisp_R, crisp_Q
