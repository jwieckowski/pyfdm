# Copyright (c) 2023 Jakub Więckowski, Andrii Shekhovtsov

import numpy as np
from functools import reduce
from pyfdm.TFN import TFN

def fuzzy(matrix, weights, normalization, bounds, isp):
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

            bounds : ndarray
                Decision problem bounds / criteria bounds. Should be two dimensional array with [min, max] value for in criterion in rows.

            isp : ndarray
                Vector of Ideal Solution Point
                
        Returns
        -------
            ndarray:
                Crisp preferences of alternatives

    """
    def algsum(a, b):
        return (a + b) - a * b

    def aggregation(d, weights, n=512, operator=algsum):
        x = np.linspace(np.min(d).a, np.max(d).c, n)
        mu_values = [w * tfn.membership_function(x) for w, tfn in zip(weights, d)]
        summed = reduce(operator, mu_values)

        cog = np.sum(x * summed) / np.sum(summed)
        return cog

    # normalized decision matrix
    if normalization is not None:
        nmatrix = normalization(matrix)
    else:
        nmatrix = matrix.copy()

    tfn_matrix = np.array([[TFN(*m) for m in row] for row in nmatrix])

    d = tfn_matrix.copy()
    for i in range(len(isp)):
        d[:, i] = np.abs((tfn_matrix[:, i] - isp[i]) / (bounds[i, 1] - bounds[i, 0]))

    res = []
    for alt in d:
        res.append(aggregation(alt, weights))

    return np.array(res)
