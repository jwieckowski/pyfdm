# Copyright (c) 2022 Jakub Więckowski

import numpy as np

def fuzzy(matrix, weights, types, normalization, distance_1, distance_2, tau):
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

            distance_1: callable
                Function used to calculate distance from fuzzy negative solution

            distance_2: callable
                Function used to calculate distance form fuzzy negative solution

            tau: float
                Threshold parameter

        Returns
        -------
            ndarray:
                Crisp preferences of alternatives

    """

    def _psi(x, tau=0.02):
        """
            Threshold function

            Parameters
            ----------
                x: float
                    Value to assess

                tau: float, default=0.02
                    Threshold parameter

            Returns
            -------
                int
                    0 if absolute value of x below threshold, otherwise 1
        """
        # tau from 0.01 to 0.05
        if np.abs(x) >= tau:
            return 1
        return 0

    # normalized decision matrix
    nmatrix = normalization(matrix, types)

    if weights.ndim == 1:
        weights = np.repeat(weights, 3).reshape((len(weights), 3))

    # weighted decision matrix
    wmatrix = nmatrix * weights

    # fuzzy negative solution
    NS = np.min(wmatrix, axis=0)

    # distances from fuzzy negative solution
    D1, D2 = np.zeros((matrix.shape[0], ), dtype=object), np.zeros(
        (matrix.shape[0], ), dtype=object)
    for i in range(matrix.shape[0]):
        D1[i] = np.sum([distance_1(wmatrix[i, j], NS[j])
                       for j in range(matrix.shape[1])])
        D2[i] = np.sum([distance_2(wmatrix[i, j], NS[j])
                       for j in range(matrix.shape[1])])

    # relative assessment matrix
    RA = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=object)
    for i in range(RA.shape[0]):
        for j in range(RA.shape[1]):
            RA[i, j] = (D1[i] - D1[j]) + \
                (_psi(D1[i] - D1[j], tau) * (D2[i] - D2[j]))

    # assessment score
    AS = np.sum(RA, axis=1)
    return AS
