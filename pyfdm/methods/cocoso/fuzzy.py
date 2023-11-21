# Copyright (c) 2023 Jakub Więckowski

import numpy as np

def fuzzy(matrix, weights, types, normalization, defuzzify, d=0.5):
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

            d: float, default=0.5
                Parameter included in the assessment score, determined by decision-maker
        Returns
        -------
            ndarray:
                Crisp preferences of alternatives

    """

    # normalized decision matrix
    nmatrix = normalization(matrix, types)
    
    if weights.ndim == 1:
        weights = np.repeat(weights, 3).reshape((len(weights), 3))

    # sum of comparability
    S = np.sum(nmatrix * weights, axis=1)

    # sum of power weights (weights order inside TFN is reversed)
    P = np.sum(nmatrix ** weights[..., ::-1], axis=1)
    
    # fuzzy evaluation score
    fa = np.array([(P[i, :] + S[i, :]) / (np.sum(P + S, axis=0)[..., ::-1]) for i in range(matrix.shape[0])])
    fb = np.array([S[i, :]/np.min(S) + P[i, :]/np.min(P) for i in range(matrix.shape[0])])
    fc = np.array([(d*S[i, :] + (1-d) * P[i, :]) / (d * np.max(S) + (1-d) * np.max(P)) for i in range(matrix.shape[0])])

    # fuzzy net assessment scores
    nfa = np.array([defuzzify(f) for f in fa])
    nfb = np.array([defuzzify(f) for f in fb])
    nfc = np.array([defuzzify(f) for f in fc])

    # # crisp assessment
    f = (nfa * nfb * nfc) * (1/3) + ( (nfa + nfb + nfc) / 3)

    return f
