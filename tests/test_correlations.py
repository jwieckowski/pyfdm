# Copyright (c) 2022 Jakub Więckowski

import numpy as np
from pyfdm.correlations import *


def test_spearman_coef():
    """
        Test veryfing correctness of the spearman correlation coefficient formula.
        Reference value: Sałabun, W., & Urbaniak, K. (2020, June). A new coefficient of rankings similarity in decision-making problems. In International Conference on Computational Science (pp. 632-645). Springer, Cham.
    """
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 1, 3, 4, 5])
    assert np.round(spearman_coef(x, y), 2) == 0.9


def test_person_coef():
    """
        Test veryfing correctness of the pearson correlation coefficient formula.
        Reference value: Sałabun, W., Karczmarczyk, A., & Wątróbski, J. (2018, November). Decision-making using the hesitant fuzzy sets COMET method: An empirical study of the electric city buses selection. In 2018 IEEE Symposium Series on Computational Intelligence (SSCI) (pp. 1485-1492). IEEE.
    """
    x = np.array([
        0.5263, 0.7368, 1.0, 0.3158, 0.6842, 0.8947,
        0.1053, 0.4737, 0.7895, 0.4211, 0.7895, 0.9474,
        0.2105, 0.5789, 0.8421, 0.0526, 0.3684, 0.7368,
        0.3158, 0.6316, 0.8947, 0.1579, 0.4737, 0.8421,
        0.0, 0.2631, 0.5789
    ])
    y = np.array([
        0.6316, 0.8421, 1.0, 0.3684, 0.7895, 1.0,
        0.1053, 0.5263, 0.9474, 0.4737, 0.9474, 1.0,
        0.2105, 0.6842, 1.0, 0.0526, 0.4211, 0.8947,
        0.3158, 0.7368, 1.0, 0.1579, 0.5789, 1.0,
        0.0, 0.2632, 0.6842
    ])
    assert np.round(pearson_coef(x, y), 3) == 0.992


def test_weighted_spearman_coef():
    """
        Test veryfing correctness of the weighted spearman correlation coefficient formula.
        Reference value: Paradowski, B., Bączkiewicz, A., & Watrąbski, J. (2021). Towards proper consumer choices-MCDM based product selection. Procedia Computer Science, 192, 1347-1358.
    """
    x = np.array([7, 11, 2, 1, 4, 9, 6, 3, 5, 10, 8, 12])
    y = np.array([7, 11, 3, 2, 4, 9, 6, 1, 5, 10, 8, 12])
    assert np.round(weighted_spearman_coef(x, y), 2) == 0.96


def test_ws_rank_similarity_coef():
    """
        Test veryfing correctness of the ws rank similarity coefficient formula.
        Reference value: Paradowski, B., Bączkiewicz, A., & Watrąbski, J. (2021). Towards proper consumer choices-MCDM based product selection. Procedia Computer Science, 192, 1347-1358.
    """
    x = np.array([7, 11, 2, 1, 4, 9, 6, 3, 5, 10, 8, 12])
    y = np.array([7, 11, 3, 2, 4, 9, 6, 1, 5, 10, 8, 12])
    assert np.round(ws_rank_similarity_coef(x, y), 2) == 0.9
