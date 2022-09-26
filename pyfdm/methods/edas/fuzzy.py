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

    def psi(a):
        """
            Threshold function

            Parameters
            ----------
                a: ndarray
                    Triangular Fuzzy Number

            Returns
            -------
                ndarray
                    0 if defuzzified TFN lower than 0, otherwise TFN
        """
        if defuzzify(a) > 0:
            return a
        return [0, 0, 0]

    # fuzzy average decision matrix
    av_matrix = np.mean(matrix, axis=0)
    k = np.array([defuzzify(a) for a in av_matrix])

    # positive and negative distances from average
    pda, nda = np.zeros((matrix.shape)), np.zeros((matrix.shape))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if types[j] == 1:
                pda[i, j] = psi(
                    matrix[i, j] - av_matrix[j][..., ::-1]) / k[j]
                nda[i, j] = psi(
                    av_matrix[j] - matrix[i, j][..., ::-1]) / k[j]
            else:
                pda[i, j] = psi(
                    av_matrix[j] - matrix[i, j][..., ::-1]) / k[j]
                nda[i, j] = psi(
                    matrix[i, j] - av_matrix[j][..., ::-1]) / k[j]

    if weights.ndim == 1:
        weights = np.repeat(weights, 3).reshape((len(weights), 3))

    # fuzzy weighted positive and negative distances
    sp = np.sum(pda * weights, axis=1)
    sn = np.sum(nda * weights, axis=1)

    # fuzzy normalized weighted positive and negative distances
    nsp = sp / np.max([defuzzify(p) for p in sp])
    nsn = 1 - (sn / np.max([defuzzify(n) for n in sn]))

    # fuzzy appraisal score
    a = (nsp + nsn) / 2
    return np.array([defuzzify(alt) for alt in a])
