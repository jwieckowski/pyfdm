# Copyright (c) 2022 Jakub Więckowski

import numpy as np
import pyfdm.methods.fuzzy_sets.tfn.distances as dist
from pyfdm.methods import f_topsis

def test_euclidean_distance():
    """
        Test veryfing correctness of the Euclidean distance formula.
        Reference value: Vidyadhar, R., Kumar, R. S., Vinodh, S., & Antony, J. (2016). Application of fuzzy logic for leanness assessment in SMEs: a case study. Journal of Engineering, Design and Technology.
    """
    x = np.array([4.42, 6, 7.64])
    y = np.array([7, 8.5, 10])
    calculated_value = dist.euclidean_distance(x, y)
    reference_value = 4.3

    assert np.round(calculated_value, 1) == reference_value


def test_weighted_euclidean_distance(): 
    """
        Test veryfing correctness of the weighted Euclidean distance formula.
        Formula: Roszkowska, E., & Wachowicz, T. (2015). Application of fuzzy TOPSIS to scoring the negotiation offers in ill-structured negotiation problems. European Journal of Operational Research, 242(3), 920-932.
        Reference value: Self-calculated empirical verification
    """
    x = np.array([3, 5, 8])
    y = np.array([2, 4, 8])
    calculated_value = dist.weighted_euclidean_distance(x, y)
    reference_value = 0.866
    assert np.round(calculated_value, 3) == reference_value

def test_hamming_distance():
    """
        Test veryfing correctness of the Hamming distance formula.
        Reference value: Talukdar, P., & Dutta, P. A Comparative Study of TOPSIS Method via Different Distance Measure.
    """
    matrix = np.array([
        [[0.50, 0.70, 0.90], [0.65, 0.80, 0.95], [0.35, 0.50, 0.65], [0.90, 1.00, 1.00], [0.35, 0.50, 0.65]],
        [[0.65, 0.80, 0.95], [0.90, 1.00, 1.00], [0.90, 1.00, 1.00], [0.90, 1.00, 1.00], [0.90, 1.00, 1.00]],
        [[0.90, 1.00, 1.00], [0.50, 0.70, 0.90], [0.65, 0.80, 0.95], [0.65, 0.80, 0.95], [0.65, 0.80, 0.95]],
    ])
    weights = np.array([[0.75, 0.87, 0.95], [0.90, 1.00, 1.00], [0.83, 0.93, 1.00], [0.90, 1.00, 1.00], [0.48, 0.64, 0.78]])
    types = np.array([1, 1, 1, 1, 1])

    topsis = f_topsis.fTOPSIS(distance=dist.hamming_distance)
    
    calculated_value = topsis(matrix, weights, types)
    reference_value = np.array([0.62, 0.82, 0.71])

    assert (np.round(calculated_value.astype(float), 2) == reference_value).all() or np.sum(np.abs(calculated_value - reference_value)) < 0.05

def test_weighted_hamming_distance():
    """
        Test veryfing correctness of the weighted Hamming distance formula.
        Reference value: Roszkowska, E., & Wachowicz, T. (2015). Application of fuzzy TOPSIS to scoring the negotiation offers in ill-structured negotiation problems. European Journal of Operational Research, 242(3), 920-932.
    """
    x = np.array([3, 5, 8])
    y = np.array([2, 4, 8])
    calculated_value = dist.weighted_hamming_distance(x, y)
    reference_value = 0.75
    assert np.round(calculated_value, 3) == reference_value

def test_vertex_distance():
    """
        Test veryfing correctness of the Vertex distance formula.
        Reference value: Roszkowska, E., & Wachowicz, T. (2015). Application of fuzzy TOPSIS to scoring the negotiation offers in ill-structured negotiation problems. European Journal of Operational Research, 242(3), 920-932.
    """
    x = np.array([3, 5, 8])
    y = np.array([2, 4, 8])
    calculated_value = dist.vertex_distance(x, y)
    reference_value = 0.82

    assert np.round(calculated_value, 2) == reference_value


def test_tran_duckstein_distance():
    """
        Test veryfing correctness of the Tran Duckstein distance formula.
        Reference value: Tran, L., & Duckstein, L. (2002). Comparison of fuzzy numbers using a fuzzy distance measure. Fuzzy sets and Systems, 130(3), 331-341.
    """
    x = np.array([0.4, 0.43, 1])
    y = np.array([0.4, 0.83, 1])
    calculated_value = dist.tran_duckstein_distance(x, y)
    reference_value = 0.187

    assert np.round(calculated_value, 3) == reference_value


def test_lr_distance():
    """
        Test veryfing correctness of the L-R distance formula.
        Reference value: Talukdar, P., & Dutta, P. A Comparative Study of TOPSIS Method via Different Distance Measure.
    """
    matrix = np.array([
        [[0.50, 0.70, 0.90], [0.65, 0.80, 0.95], [0.35, 0.50, 0.65], [0.90, 1.00, 1.00], [0.35, 0.50, 0.65]],
        [[0.65, 0.80, 0.95], [0.90, 1.00, 1.00], [0.90, 1.00, 1.00], [0.90, 1.00, 1.00], [0.90, 1.00, 1.00]],
        [[0.90, 1.00, 1.00], [0.50, 0.70, 0.90], [0.65, 0.80, 0.95], [0.65, 0.80, 0.95], [0.65, 0.80, 0.95]],
    ])
    weights = np.array([[0.75, 0.87, 0.95], [0.90, 1.00, 1.00], [0.83, 0.93, 1.00], [0.90, 1.00, 1.00], [0.48, 0.64, 0.78]])
    types = np.array([1, 1, 1, 1, 1])

    topsis = f_topsis.fTOPSIS(distance=dist.lr_distance)
    
    calculated_value = topsis(matrix, weights, types)
    reference_value = np.array([0.74, 0.96, 0.90])

    assert (np.round(calculated_value.astype(float), 2) == reference_value).all() or np.sum(np.abs(calculated_value - reference_value)) < 0.05


def test_mahdavi_distance():
    """
        Test veryfing correctness of the Mahdavi distance formula.
        Formula: Wang, H., Lu, X., Du, Y., Zhang, C., Sadiq, R., & Deng, Y. (2017). Fault tree analysis based on TOPSIS and triangular fuzzy number. International journal of system assurance engineering and management, 8(4), 2064-2070.
        Reference value: Self-calculated empirical verification
    """
    # https://link.springer.com/article/10.1007/s13198-014-0323-5#Equ20
    x = np.array([0.92, 0.96, 1.0])
    y = np.array([0.55, 0.6643, 0.7786])
    calculated_value = dist.mahdavi_distance(x, y)
    # reference_value = 0.2278
    reference_value = 0.2988

    assert np.round(calculated_value, 4) == reference_value
