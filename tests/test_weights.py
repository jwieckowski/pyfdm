# Copyright (c) 2022 Jakub Więckowski

import numpy as np
from pyfdm.weights import *


def test_equal_weights():
    """
        Test veryfing correctness of the equal weights methods.
        Each weight should have the same value for all criteria and for each element of the Triangular Fuzzy Number
    """

    matrix = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4]
    ])

    reference_weights = np.array([
        [0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25]
    ])

    calculated_weights = np.round(equal_weights(matrix), 3)
    assert (calculated_weights == reference_weights).all()


def test_shannon_entropy_weights():
    """
        Test veryfing correctness of the shannon entropy weights methods.
        Reference value: Kacprzak, D. (2017). Objective weights based on ordered fuzzy numbers for fuzzy multiple criteria decision-making methods. Entropy, 19(7), 373
    """

    matrix = np.array([
        [[0.313, 0.37, 0.45], [0.3, 0.278, 0.269], [0.25, 0.257, 0.256], [
            0.346, 0.303, 0.27], [0.313, 0.292, 0.281], [0.417, 0.35, 0.321]],
        [[0.219, 0.185, 0.150], [0.1, 0.167, 0.192], [0.179, 0.2, 0.231], [
            0.115, 0.152, 0.189], [0.188, 0.208, 0.219], [0.25, 0.25, 0.25]],
        [[0.313, 0.333, 0.350], [0.5, 0.389, 0.346], [0.25, 0.257, 0.256], [
            0.269, 0.273, 0.27], [0.188, 0.208, 0.219], [0.083, 0.15, 0.179]],
        [[0.156, 0.111, 0.05], [0.1, 0.167, 0.192], [0.321, 0.286, 0.256], [
            0.269, 0.273, 0.27], [0.313, 0.292, 0.281], [0.25, 0.25, 0.25]]
    ])

    reference_weights = np.array([
        [0.075, 0.378, 0.756],
        [0.443, 0.257, 0.108],
        [0.042, 0.032, 0.005],
        [0.130, 0.115, 0.036],
        [0.063, 0.056, 0.026],
        [0.248, 0.162, 0.069]
    ])

    calculated_weights = np.round(shannon_entropy_weights(matrix), 3)
    assert (calculated_weights == reference_weights).all()


def test_standard_deviation_weights():
    """
        Test veryfing correctness of the standard deviation weights methods.
        Formula: Wang, Y. M., & Luo, Y. (2010). Integration of correlations with standard deviations for determining attribute weights in multiple attribute decision making. Mathematical and Computer Modelling, 51(1-2), 1-12.
        Reference value: Self-calculated empirical verification
    """

    matrix = np.array([
        [[3, 5, 7], [1, 4, 5], [2, 6, 8]],
        [[1, 2, 6], [3, 4, 5], [6, 8, 9]],
        [[1, 1, 4], [2, 3, 6], [6, 7, 8]]
    ])

    reference_weights = np.array([
        [0.259, 0.569, 0.569],
        [0.224, 0.158, 0.215],
        [0.517, 0.273, 0.215]
    ])

    calculated_weights = np.round(standard_deviation_weights(matrix), 3)

    assert (calculated_weights == reference_weights).all()


def test_variance_weights():
    """
        Test veryfing correctness of the variance weights methods.
        Formula: Bikmukhamedov, R., Yeryomin, Y., & Seitz, J. (2016, July). Evaluation of MCDA-based handover algorithms for mobile networks. In 2016 Eighth International Conference on Ubiquitous and Future Networks (ICUFN) (pp. 810-815). IEEE.
        Reference value: Self-calculated empirical verification
    """

    matrix = np.array([
        [[3, 5, 7], [1, 4, 5], [2, 6, 8]],
        [[1, 2, 6], [3, 4, 5], [6, 8, 9]],
        [[1, 1, 4], [2, 3, 6], [6, 7, 8]]
    ])

    reference_weights = np.array([
        [0.174, 0.765, 0.778],
        [0.130, 0.059, 0.111],
        [0.696, 0.176, 0.111]
    ])

    calculated_weights = np.round(variance_weights(matrix), 3)
    
    assert (calculated_weights == reference_weights).all()
