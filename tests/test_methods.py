# Copyright (c) 2022 Jakub Więckowski

import numpy as np
from pyfdm.methods import *


def test_fARAS():
    """
        Test verifying correctness of the fuzzy ARAS method combined with Triangular Fuzzy Number
        Reference value: Fu, Y. K., Wu, C. J., & Liao, C. N. (2021). Selection of in-flight duty-free product suppliers using a combination fuzzy AHP, fuzzy ARAS, and MSGP methods. Mathematical Problems in Engineering, 2021.
    """

    matrix = np.array([
        [[0.3, 0.54, 1], [0.3, 0.58, 0.9], [0.4, 0.63, 0.9],
            [0.3, 0.49, 0.8], [0.4, 0.67, 1]],
        [[0.3, 0.56, 0.9], [0.3, 0.49, 0.8], [0.4, 0.63, 0.8],
            [0.4, 0.59, 0.98], [0.4, 0.59, 0.9]],
        [[0.3, 0.52, 0.8], [0.5, 0.7, 1], [0.3, 0.54, 0.9],
            [0.3, 0.34, 0.9], [0.4, 0.59, 0.9]],
        [[0.4, 0.59, 0.9], [0.4, 0.64, 1], [0.3, 0.73, 1],
            [0.4, 0.63, 0.9], [0.3, 0.58, 1]],
        [[0.5, 0.77, 1], [0.4, 0.63, 0.9], [0.3, 0.6, 0.9],
            [0.4, 0.64, 1], [0.4, 0.63, 1]],
    ])

    weights = np.array([[0.4, 0.71, 1], [0.4, 0.61, 0.9], [
        0.5, 0.73, 1], [0.3, 0.52, 0.9], [0.2, 0.63, 1]])
    types = np.array([1, 1, 1, 1, 1])

    f_aras = fARAS()

    calculated_result = f_aras(matrix, weights, types)
    reference_result = np.array([0.76, 0.73, 0.74, 0.8, 0.81])
    assert (np.round(calculated_result.astype(float), 2) == reference_result).all()
    assert (f_aras.rank() == [3, 5, 4, 2, 1]).all()


def test_fCODAS():
    """
        Test verifying correctness of the fuzzy CODAS method combined with Triangular Fuzzy Number
        Decision matrix: Narang, M., Joshi, M. C., & Pal, A. K. (2021). A hybrid fuzzy COPRAS-base-criterion method for multi-criteria decision making. Soft Computing, 25(13), 8391-8399.
        Reference value: Self-calculated empirical verification
    """

    matrix = np.array([
        [[3, 4, 5],[4, 5, 6],[8, 9, 9]],
        [[6, 7, 8],[4, 5, 6],[1, 2, 3]],
        [[5, 6, 7],[2, 3, 4],[3, 4, 5]],
        [[8, 9, 9],[2, 3, 4],[2, 3, 4]],
        [[7, 8, 9],[7, 8, 9],[5, 6, 7]],
    ])

    weights = np.array([0.394, 0.084, 0.522])
    types = np.array([-1, 1, 1])
    f_codas = fCODAS()

    calculated_result = f_codas(matrix, weights, types)
    reference_result = np.array([8.65, -4.15, -0.72, -5.06, 1.28])

    assert (np.round(calculated_result.astype(float), 2) == reference_result).all()
    assert (f_codas.rank() == [1, 4, 3, 5, 2]).all()

def test_fCOPRAS():
    """
        Test verifying correctness of the fuzzy COPRAS method combined with Triangular Fuzzy Number
        Reference value: Narang, M., Joshi, M. C., & Pal, A. K. (2021). A hybrid fuzzy COPRAS-base-criterion method for multi-criteria decision making. Soft Computing, 25(13), 8391-8399.
    """

    matrix = np.array([
        [[3, 4, 5],[4, 5, 6],[8, 9, 9]],
        [[6, 7, 8],[4, 5, 6],[1, 2, 3]],
        [[5, 6, 7],[2, 3, 4],[3, 4, 5]],
        [[8, 9, 9],[2, 3, 4],[2, 3, 4]],
        [[7, 8, 9],[7, 8, 9],[5, 6, 7]],
    ])

    weights = np.array([0.394, 0.084, 0.522])
    types = np.array([-1, 1, 1])
    
    f_copras = fCOPRAS()

    calculated_result = f_copras(matrix, weights, types)
    reference_result = np.array([1.014, 0.508, 0.590, 0.470, 0.642])

    assert (calculated_result == reference_result).all() or np.sum(calculated_result - reference_result) < 0.15
    assert (f_copras.rank() == [1, 4, 3, 5, 2]).all()

def test_fEDAS():
    """
        Test verifying correctness of the fuzzy EDAS method combined with Triangular Fuzzy Number
        Reference value: Yılmaz, M., & Atan, T. (2021). Hospital site selection using fuzzy EDAS method: case study application for districts of Istanbul. Journal of Intelligent & Fuzzy Systems, (Preprint), 1-12.
    """
    matrix = np.array([
        [
            [0.500, 0.700, 0.867], [0.700, 0.867, 0.967], [0.767, 0.933, 1.000], [0.700, 0.867, 0.967],
            [0.300, 0.500, 0.700], [0.767, 0.933, 1.000], [0.300, 0.500, 0.700], [0.767, 0.933, 1.000],
            [0.633, 0.800, 0.933], [0.567, 0.767, 0.900], [0.500, 0.667, 0.800], [0.900, 1.000, 1.000],
            [0.500, 0.700, 0.867], [0.700, 0.867, 0.967], [0.833, 0.967, 1.000], [0.700, 0.867, 0.967],
            [0.700, 0.867, 0.967]
        ],
        [
            [0.767, 0.900, 0.967], [0.767, 0.933, 1.000], [0.900, 1.000, 1.000], [0.633, 0.833, 0.967],
            [0.200, 0.367, 0.567], [0.700, 0.867, 0.967], [0.367, 0.567, 0.733], [0.833, 0.967, 1.000], 
            [0.567, 0.733, 0.867], [0.767, 0.900, 0.967], [0.500, 0.700, 0.867], [0.833, 0.967, 1.000], 
            [0.633, 0.833, 0.967], [0.633, 0.833, 0.967], [0.767, 0.933, 1.000], [0.700, 0.867, 0.967], 
            [0.700, 0.900, 1.000]
        ],
        [
            [0.767, 0.933, 1.000], [0.833, 0.967, 1.000], [0.833, 0.967, 1.000], [0.567, 0.767, 0.933],
            [0.200, 0.367, 0.567], [0.700, 0.867, 0.967], [0.433, 0.633, 0.800], [0.900, 1.000, 1.000],
            [0.700, 0.867, 0.967], [0.633, 0.800, 0.900], [0.433, 0.633, 0.800], [0.767, 0.933, 1.000],
            [0.500, 0.700, 0.900], [0.700, 0.867, 0.967], [0.833, 0.967, 1.000], [0.767, 0.933, 1.000],
            [0.633, 0.833, 0.967]
        ]
    ])

    weights = np.array([
        [0.700, 0.867, 0.967],
        [0.767, 0.933, 1.000],
        [0.900, 1.000, 1.000],
        [0.767, 0.933, 1.000],
        [0.633, 0.833, 0.967],
        [0.833, 0.967, 1.000],
        [0.300, 0.500, 0.700],
        [0.900, 1.000, 1.000],
        [0.633, 0.833, 0.967],
        [0.833, 0.967, 1.000],
        [0.433, 0.633, 0.800],
        [0.767, 0.933, 1.000],
        [0.633, 0.833, 0.967],
        [0.633, 0.833, 0.967],
        [0.833, 0.967, 1.000],
        [0.633, 0.833, 0.967],
        [0.833, 0.967, 1.000],
    ])
    types = np.array([1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    f_edas = fEDAS()

    calculated_result = f_edas(matrix, weights, types)
    reference_result = np.array([0.212, 0.514, 0.746])

    assert (calculated_result == reference_result).all() or np.sum(calculated_result - reference_result) < 0.3
    assert (f_edas.rank() == [3, 1, 2]).all()


def test_fMABAC():
    """
        Test verifying correctness of the fuzzy MABAC method combined with Triangular Fuzzy Number
        Reference value: Bozanic, D., Tešić, D., & Milićević, J. (2018). A hybrid fuzzy AHP-MABAC model: Application in the Serbian Army–The selection of the location for deep wading as a technique of crossing the river by tanks. Decision Making: Applications in Management and Engineering, 1(1), 143-164.
    """

    matrix = np.array([
        [[115, 120, 126], [3, 4, 5], [4, 5, 5], [0.9, 1.1, 1.3],
            [2, 3, 4], [1.3, 1.5, 1.7], [1, 1, 2]],
        [[134, 140, 147], [4, 5, 5], [2, 3, 4], [0.7, 0.9, 1.2],
            [1, 1, 2], [1.1, 1.3, 1.5], [3, 4, 5]],
        [[105, 110, 115], [4, 5, 5], [3, 4, 5], [1.05, 1.2, 1.4],
            [4, 5, 5, ], [1.4, 1.6, 1.8], [2, 3, 4]],
        [[120, 125, 130], [2, 3, 4], [1, 1, 2], [0.8, 1, 1.2],
            [3, 4, 5], [1.3, 1.5, 1.7], [1, 1, 2]],
        [[153, 160, 170], [3, 4, 5], [4, 5, 5], [0.6, 0.7, 0.8],
            [4, 5, 5], [1, 1.2, 1.4], [3, 4, 5]],
        [[114, 118, 126], [2, 3, 4], [2, 3, 4], [1.1, 1.15, 1.25],
            [3, 4, 5], [1.3, 1.5, 1.7], [2, 3, 4]],
    ])

    weights = np.array([0.243, 0.159, 0.182, 0.097, 0.125, 0.071, 0.123])
    types = np.array([-1, 1, -1, -1, 1, -1, 1])

    f_mabac = fMABAC()

    calculated_result = f_mabac(matrix, weights, types)
    reference_result = np.array([-0.071, 0.032, 0.113, 0.053, -0.014, 0.029])

    assert (np.round(calculated_result, 3) == reference_result).all() or np.sum(np.abs(calculated_result - reference_result)) < 0.05
    assert (f_mabac.rank() == [6, 3, 1, 2, 5, 4]).all()

def test_fMAIRCA():
    """
        Test verifying correctness of the fuzzy MAIRCA method combined with Triangular Fuzzy Number
        Reference value: Boral, S., Howard, I., Chaturvedi, S. K., McKee, K., & Naikan, V. N. A. (2020). An integrated approach for fuzzy failure modes and effects analysis using fuzzy AHP and fuzzy MAIRCA. Engineering Failure Analysis, 108, 104195.
    """

    matrix = np.array([
        [[2.33, 4.33, 6.33], [4.33, 6.33, 8.33], [6.33, 8.33, 9.66]],
        [[0.66, 2.33, 4.33], [8.33, 9.66, 10], [0.66, 2.33, 4.33]],
        [[0.66, 2.33, 4.33], [8.33, 9.66, 10], [0.33, 1.33, 3]],
        [[1.66, 3.66, 5.66], [4.33, 6.33, 8.33], [6.33, 8.33, 9.66]],
        [[2.33, 4.33, 6.33], [5.66, 7.66, 9.33], [7.66, 9.33, 10]],
        [[4.33, 6.33, 8.33], [5.66, 7.66, 9.33], [1.66, 3.66, 5.66]],
        [[0.33, 1.33, 3], [9, 10, 10], [0.33, 1.33, 3]],
        [[0, 0.33, 1.66], [0, 0, 1], [0, 0, 1]],
    ])

    weights = np.array(
        [[0.293, 0.388, 0.565], [0.203, 0.267, 0.386], [0.234, 0.345, 0.420]])

    f_mairca = fMAIRCA()

    calculated_result = f_mairca(matrix, weights, types=None)
    reference_result = np.array([0.0986, 0.1103, 0.1121, 0.1104, 0.0963, 0.0986, 0.1147, 0.1291])
    
    assert (np.round(calculated_result, 4) == reference_result).all() or np.sum(np.abs(calculated_result - reference_result)) < 0.05
    assert (f_mairca.rank() == [7, 4, 3, 5, 8, 6, 2, 1]).all()

def test_fMOORA():
    """
        Test verifying correctness of the fuzzy MOORA method combined with Triangular Fuzzy Number
        Reference value: Karande, P., & Chakraborty, S. (2012). A Fuzzy-MOORA approach for ERP system selection. Decision Science Letters, 1(1), 11-21.
    """

    matrix = np.array([
        [[0.6, 0.8, 1], [0.6, 0.8, 1], [0.4, 0.6, 0.8], [0.2, 0.4, 0.6], [0.8, 1, 1]],
        [[0.4, 0.6, 0.8], [0.6, 0.8, 1], [0.4, 0.6, 0.8],
            [0.8, 1, 1], [0.2, 0.4, 0.6]],
        [[0.8, 1, 1], [0.4, 0.6, 0.8], [0.6, 0.8, 1], [0, 0.2, 0.4], [0, 0.2, 0.4]],
        [[0.2, 0.4, 0.6], [0.4, 0.6, 0.8], [
            0.4, 0.6, 0.8], [0.6, 0.8, 1], [0.4, 0.6, 0.8]]
    ])

    weights = np.array([0.364, 0.271, 0.203, 0.094, 0.068])
    types = np.array([-1, 1, 1, 1, 1])

    f_moora = fMOORA()

    calculated_result = f_moora(matrix, weights, types)
    reference_result = np.array([0.0713, 0.1034, 0.0355, 0.1113])

    assert (calculated_result == reference_result).all() or np.sum(np.abs(calculated_result - reference_result)) < 0.05
    assert (f_moora.rank() == [3, 2, 4, 1]).all()

def test_fOCRA():
    """
        Test verifying correctness of the fuzzy OCRA method combined with Triangular Fuzzy Number
        Reference value: ULUTAŞ, A. (2019). Supplier selection by using a fuzzy integrated model for a textile company. Engineering Economics, 30(5), 579-590.
    """

    matrix = np.array([
        [[4.75, 5.75, 6.75], [5.5, 6.5, 7.5], [5, 6, 7], [3.75, 4.75, 5.75],
            [1.25, 2.25, 3.25], [1, 2, 3], [4.75, 5.75, 6.75], [5, 6, 7]],
        [[3.75, 4.75, 5.75], [4.5, 5.5, 6.5], [4.25, 5.25, 6.25], [5.75, 6.75, 7.75], [
            1.5, 2.5, 3.5], [1.75, 2.75, 3.75], [4.5, 5.5, 6.5], [5.25, 6.25, 7.25]],
        [[4.5, 5.5, 6.5], [4.75, 5.75, 6.75], [4.75, 5.75, 6.75], [4.25, 5.25, 6.25], [
            1, 2, 3], [1.75, 2.75, 3.75], [4, 5, 6], [5.75, 6.75, 7.75]],
        [[3, 4, 5], [4.25, 5.25, 6.25], [3.5, 4.5, 5.5], [3, 4, 5], [2, 3, 4],
            [1.5, 2.5, 3.5], [4.75, 5.75, 6.75], [5.25, 6.25, 7.25]],
        [[5, 6, 7], [3, 4, 5], [4.5, 5.5, 6.5], [3.25, 4.25, 5.25], [
            1.75, 2.75, 3.75], [1.25, 2.25, 3.25], [4.5, 5.5, 6.5], [5.5, 6.5, 7.5]]
    ])

    weights = np.array([0.19, 0.182, 0.122, 0.091, 0.198, 0.074, 0.078, 0.065])
    types = np.array([1, 1, 1, 1, -1, -1, 1, 1])

    f_ocra = fOCRA()

    calculated_result = f_ocra(matrix, weights, types)
    reference_result = np.array([0.317, 0.181, 0.266, 0, 0.115])
    
    assert (np.round(calculated_result.astype(float), 3) == reference_result).all() or np.sum(np.abs(np.round(calculated_result.astype(float), 3) - reference_result)) < 0.05 
    assert (f_ocra.rank() == [1, 3, 2, 5, 4]).all()

def test_fTOPSIS():
    """
        Test verifying correctness of the fuzzy TOPSIS method combined with Triangular Fuzzy Number
        Reference value: Chen, C. T. (2000). Extensions of the TOPSIS for group decision-making under fuzzy environment. Fuzzy sets and systems, 114(1), 1-9.
    """

    matrix = np.array([
        [[5.7, 7.7, 9.3], [5, 7, 9], [5.7, 7.7, 9], [8.33, 9.67, 10], [3, 5, 7]],
        [[6.3, 8.3, 9.7], [9, 10, 10], [8.3, 9.7, 10], [9, 10, 10], [7, 9, 10]],
        [[6.3, 8, 9], [7, 9, 10], [7, 9, 10], [7, 9, 10], [6.3, 8.3, 9.7]]
    ])

    weights = np.array([[0.7, 0.9, 1], [0.9, 1, 1],
                        [0.77, 0.93, 1], [0.9, 1, 1], [0.43, 0.63, 0.83]])

    types = np.array([1, 1, 1, 1, 1])

    f_topsis = fTOPSIS()

    calculated_result = f_topsis(matrix, weights, types)
    reference_result = np.array([0.62, 0.77, 0.71])

    assert (np.round(calculated_result.astype(float), 2) == reference_result).all() or np.sum(np.abs(calculated_result - reference_result)) < 0.05
    assert (f_topsis.rank() == [3, 1, 2]).all()

def test_fVIKOR():
    """
        Test verifying correctness of the fuzzy VIKOR method combined with Triangular Fuzzy Number
        Reference value: Opricovic, S. (2007). A fuzzy compromise solution for multicriteria problems. International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, 15(03), 363-380.
    """

    types = np.array([1, -1, 1, 1, -1, -1, -1, 1])

    matrix = [
        [[3800, 4184.3, 4184.3], [4300, 5211.9, 5211.9], [4500, 5021.3, 5021.3], [
            5000, 5566.1, 5566.1], [4600, 5060.5, 5060.5], [4000, 4317.9, 4317.9]],
        [[2914.0, 2914.0, 3100], [3630.0, 3630.0, 3700], [3920.5, 3920.5, 4500], [
            3957.9, 3957.9, 4200], [3293.5, 3293.5, 3500], [2925.9, 2925.9, 3200]],
        [[350, 407.2, 440], [420, 501.7, 550], [480, 504.0, 510], [
            540, 559.5, 580], [500, 514.1, 520], [420, 432.8, 450]],
        [[250, 251.0, 260], [300, 308.3, 340], [275, 278.6, 280], [
            330, 335.3, 340], [270, 284.2, 300], [230, 239.3, 240]],
        [[195, 195, 195], [282, 282, 282], [12, 12, 12],
            [167, 167, 167], [69, 69, 69], [12, 12, 12]],
        [[244, 244, 244], [346, 346, 346], [56, 56, 56],
            [268, 268, 268], [90, 90, 90], [55, 55, 55]],
        [[15, 15, 15], [21, 21, 21], [3, 3, 3],
            [16, 16, 16], [7, 7, 7], [3, 3, 3]],
        [[2.4, 2.41, 2.5], [1.2, 1.41, 1.41], [4.3, 4.42, 4.45],
            [3.3, 3.36, 3.9], [4.0, 4.04, 4.2], [4.3, 4.36, 4.5]]
    ]

    weights = np.asarray([[1, 1.5, 4]]*4 + [[1, 1, 1]]*4)
    matrix = np.transpose(np.asarray(matrix), (1, 0, 2))

    f_vikor = fVIKOR()

    calculated_result = f_vikor(matrix, weights, types, v=0.5625)
    reference_result = np.array([
        [7.542, 7.078, 4.184, 4.189, 3.560, 4.747],
        [1.756, 1.467, 1.605, 1.487, 1.053, 1.859],
        [0.254, 0.200, 0.095, 0.081, 0.003, 0.148]
    ])

    assert (np.round(calculated_result[0], 3) == reference_result[0]).all()
    assert (np.round(calculated_result[1], 3) == reference_result[1]).all()
    assert (np.round(calculated_result[2], 3) == reference_result[2]).all()

    ranks = f_vikor.rank()
    assert (ranks[0] == [6, 5, 2, 3, 1, 4]).all()
    assert (ranks[1] == [5, 2, 4, 3, 1, 6]).all()
    assert (ranks[2] == [6, 5, 3, 2, 1, 4]).all()