# Copyright (c) 2022 Jakub Więckowski

import numpy as np
from pyfdm.helpers import *


def test_rank():
    """
        Test veryfing correctness of the rank method.
        Reference value: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html
    """
    preferences = np.array([0, 2, 3, 2])
    calculated_rank = rank(preferences)
    reference_rank = np.array([1, 2.5, 4, 2.5])
    assert (rank(calculated_rank) == reference_rank).all()


def test_generate_fuzzy_matrix():
    """
        Test veryfing correctness of the random generate fuzzy matrix method.
    """

    matrix_1 = generate_fuzzy_matrix(3, 3)
    matrix_2 = generate_fuzzy_matrix(4, 5, 3, 10)

    assert matrix_1.shape[0] == 3
    assert matrix_1.shape[1] == 3
    assert matrix_1.shape[2] == 3
    assert np.min(matrix_1) >= 0
    assert np.max(matrix_1) <= 1
    assert np.min(matrix_2) >= 3
    assert np.max(matrix_2) <= 10
