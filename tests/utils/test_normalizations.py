# Copyright (c) 2022-2023 Jakub Więckowski

import numpy as np
import pyfdm.methods.utils.normalizations as norms


def test_sum_normalization():
    """
        Test veryfing correctness of the sum normalization formula.
        Reference value: Fu, Y. K., Wu, C. J., & Liao, C. N. (2021). Selection of in-flight duty-free product suppliers using a combination fuzzy AHP, fuzzy ARAS, and MSGP methods. Mathematical Problems in Engineering, 2021.
    """
    matrix = np.array([
        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
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

    types = np.array([1, 1, 1, 1, 1])

    calculated_value = norms.sum_normalization(matrix, types)
    reference_value = np.array([0.05, 0.14, 0.31])

    assert (np.round(calculated_value[2, 0].astype(float), 2)
            == reference_value).all()


def test_max_normalization():
    """
        Test veryfing correctness of the max normalization formula.
        Formula: Panchal, D., Chatterjee, P., Shukla, R. K., Choudhury, T., & Tamosaitiene, J. (2017). Integrated Fuzzy AHP-Codas Framework for Maintenance Decision in Urea Fertilizer Industry. Economic Computation & Economic Cybernetics Studies & Research, 51(3).
        Reference value: Self-calculated empirical verification
    """
    matrix = np.array([
        [[3, 5, 7], [1, 4, 5], [2, 6, 8]],
        [[1, 2, 6], [3, 4, 5], [6, 8, 9]],
        [[1, 1, 4], [2, 3, 6], [6, 7, 8]]
    ])

    types = np.array([1, -1, 1])

    calculated_value = norms.max_normalization(matrix, types)
    reference_value = np.array([1, 1, 1])   

    assert (np.round(calculated_value[0, 0].astype(float), 2)
            == reference_value).all()


def test_linear_normalization():
    """
        Test veryfing correctness of the linear normalization formula.
        Reference value: Chen, C. T. (2000). Extensions of the TOPSIS for group decision-making under fuzzy environment. Fuzzy sets and systems, 114(1), 1-9.
    """
    matrix = np.array([
        [(5.7, 7.7, 9.3), (5, 7, 9), (5.7, 7.7, 9), (8.33, 9.67, 10), (3, 5, 7)],
        [(6.3, 8.3, 9.7), (9, 10, 10), (8.3, 9.7, 10), (9, 10, 10), (7, 9, 10)],
        [(6.3, 8, 9), (7, 9, 10), (7, 9, 10), (7, 9, 10), (6.3, 8.3, 9.7)]
    ])

    types = np.array([1, 1, 1, 1, 1])

    calculated_value = norms.linear_normalization(matrix, types)
    reference_value = np.array([0.59, 0.79, 0.96])

    assert (np.round(calculated_value[0, 0].astype(float), 2)
            == reference_value).all()

def test_minmax_normalization():
    """
        Test veryfing correctness of the Min-Max normalization formula.
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

    types = np.array([-1, 1, 1, 1, 1, -1, 1])
    calculated_value = norms.minmax_normalization(matrix, types)
    reference_value = np.array([0.68, 0.77, 0.85])

    assert (np.round(calculated_value[0, 0].astype(float), 2)
            == reference_value).all()


def test_vector_normalization():
    """
        Test veryfing correctness of the vector normalization formula.
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

    types = np.array([-1, 1, 1, 1, 1])

    calculated_value = norms.vector_normalization(matrix, types)
    reference_value = np.array([0.239, 0.318, 0.398])

    assert (np.round(calculated_value[0, 1].astype(
        float), 3) == reference_value).all()

def test_saw_normalization():
    """
        Test veryfing correctness of the saw normalization formula.
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
    weights = np.repeat(weights, 3).reshape((len(weights), 3))
    types = np.array([-1, 1, 1])

    calculated_value = norms.saw_normalization(matrix, types) * weights
    reference_value = np.array([0.065, 0.078, 0.084])

    assert (np.round(calculated_value[4, 1].astype(
        float), 3) == reference_value).all() or np.sum(np.abs(calculated_value[4,1] - reference_value)) < 0.05

def test_waspas_normalization():
    """
        Test veryfing correctness of the WASPAS normalization formula.
        Reference value: Turskis, Z., Zavadskas, E. K., Antuchevičienė, J., & Kosareva, N. (2015). A hybrid model based on fuzzy AHP and fuzzy WASPAS for construction site selection.
    """
    matrix = np.array([
        [[0.21,0.28,0.35],[0.5,0.6,0.7],[0.6,0.7,0.8],[0.6,0.7,0.8],[0.6,0.7,0.8]],
        [[0.16,0.20,0.23],[0.6,0.7,0.8],[0.6,0.7,0.8],[0.8,0.9,1.0],[0.5,0.6,0.7]],
        [[0.14,0.16,0.17],[0.8,0.9,1.0],[0.5,0.6,0.7],[0.6,0.7,0.8],[0.6,0.7,0.8]],
        [[0.09,0.12,0.17],[0.5,0.6,0.7],[0.6,0.7,0.8],[0.5,0.6,0.7],[0.4,0.5,0.6]],
        [[0.07,0.08,0.12],[0.8,0.9,1.0],[0.7,0.8,0.9],[0.6,0.7,0.8],[0.5,0.6,0.7]],
        [[0.05,0.06,0.09],[0.5,0.6,0.7],[0.8,0.9,1.0],[0.6,0.7,0.8],[0.8,0.9,1.0]],
        [[0.03,0.05,0.07],[0.4,0.5,0.6],[0.5,0.6,0.7],[0.8,0.9,1.0],[0.7,0.8,0.9]],
        [[0.01,0.03,0.06],[0.5,0.6,0.7],[0.4,0.5,0.6],[0.4,0.5,0.6],[0.5,0.6,0.7]]
    ])

    M = []
    for j in range(matrix.shape[1]):
        M.append([matrix[i, j] for i in range(matrix.shape[0])])
    M = np.array(M)
    matrix = M[1:]
    
    types = np.array([1, 1, 1, 1, 1, 1, 1, 1])

    calculated_value = norms.waspas_normalization(matrix, types)
    reference_value = np.array([0.625, 0.750, 0.875])

    assert (np.round(calculated_value[0, 0].astype(float), 3) == reference_value).all() or np.sum(np.abs(calculated_value[0, 0] - reference_value)) < 0.05

def test_sqrt_normalization():
    """
        Test veryfing correctness of the SQRT normalization formula.
        Formula: Kizielewicz, B., & Bączkiewicz, A. (2021). Comparison of Fuzzy TOPSIS, Fuzzy VIKOR, Fuzzy WASPAS and Fuzzy MMOORA methods in the housing selection problem. Procedia Computer Science, 192, 4578-4591.
        Reference value: Self-calculated empirical verification
    """
    matrix = np.array([
        [[3, 4, 5],[4, 5, 6],[8, 9, 9]],
        [[6, 7, 8],[4, 5, 6],[1, 2, 3]],
        [[5, 6, 7],[2, 3, 4],[3, 4, 5]],
        [[8, 9, 9],[2, 3, 4],[2, 3, 4]],
        [[7, 8, 9],[7, 8, 9],[5, 6, 7]],
    ])
    
    calculated_value = norms.sqrt_normalization(matrix)
    reference_value = np.array([0.192, 0.257, 0.321])

    assert (np.round(calculated_value[0, 0].astype(float), 3) == reference_value).all() or np.sum(np.abs(calculated_value[0, 0] - reference_value)) < 0.05

def test_cocoso_normalization():
    """
        Test veryfing correctness of the COCOSO normalization formula.
        Reference value: Ulutaş, A., Popovic, G., Radanov, P., Stanujkic, D., & Karabasevic, D. (2021). A new hybrid fuzzy PSI-PIPRECIA-CoCoSo MCDM based approach to solving the transportation company selection problem. Technological and Economic Development of Economy, 27(5), 1227-1249.
    """
    matrix = np.array([
        [[0.193, 0.408, 0.612],[0.572, 0.774, 0.939],[0.500, 0.700, 0.900],[0.500, 0.700, 0.900], [0.193, 0.408, 0.612],[0.332, 0.535, 0.736],[0.368, 0.572, 0.774]],
        [[0.300, 0.500, 0.700],[0.572, 0.774, 0.939],[0.408, 0.612, 0.814],[0.100, 0.300, 0.500], [0.368, 0.572, 0.774], [0.500, 0.700, 0.900],[0.612, 0.814, 0.959]],
        [[0.408, 0.612, 0.814],[0.572, 0.774, 0.939],[0.612, 0.814, 0.959],[0.856, 0.979, 1.000], [0.612, 0.814, 0.959], [0.856, 0.979, 1.000],[0.814, 0.959, 1.000]],
        [[0.125, 0.332, 0.535],[0.241, 0.451, 0.654],[0.612, 0.814, 0.959],[0.100, 0.300, 0.500], [0.368, 0.572, 0.774],[0.572, 0.774, 0.939],[0.612, 0.814, 0.959]],
        [[0.125, 0.332, 0.535],[0.241, 0.451, 0.654],[0.572, 0.774, 0.939],[0.535, 0.736, 0.919], [0.408, 0.612, 0.814],[0.814, 0.959, 1.000],[0.612, 0.814, 0.959]],
    ])

    types = np.array([-1, 1, 1, 1, -1, 1, 1])
    
    calculated_value = norms.cocoso_normalization(matrix, types)
    reference_value = np.array([0.293, 0.589, 0.901])

    assert (np.round(calculated_value[0, 0].astype(float), 3) == reference_value).all() or np.sum(np.abs(calculated_value[0, 0] - reference_value)) < 0.05