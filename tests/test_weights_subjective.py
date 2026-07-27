# Copyright (c) 2026 Jakub Więckowski

import numpy as np
from pyfdm.weights.subjective import *


def test_fAHP_weights():
    """
        Test verifying correctness of the fuzzy AHP methods.
        Reference value: Sun, C. C. (2010). A performance evaluation model by integrating fuzzy AHP and fuzzy TOPSIS methods. Expert systems with applications, 37(12), 7745-7754.
    """

    f_ahp = fAHP()
    comparison_matrix = np.array([
        [
            [1, 1, 1],
            [0.88, 1.14, 1.37],
            [1.21, 1.49, 1.74],
            [0.87, 0.98, 1.07],
            [2.14, 2.93, 3.79],
            [1.06, 1.28, 1.55]
        ],
        [
            [0.73, 0.88, 1.14],
            [1, 1, 1],
            [1.14, 1.55, 1.91],
            [1.76, 1.94, 2.09],
            [2.65, 3.36, 3.98],
            [2.14, 2.70, 3.19]
        ],
        [
            [0.58, 0.67, 0.83],
            [0.52, 0.64, 0.88],
            [1, 1, 1],
            [1.40, 1.63, 1.93],
            [1.56, 2.22, 2.91],
            [1.67, 2.13, 2.56]
        ],
        [
            [0.93, 1.02, 1.15],
            [0.48, 0.52, 0.57],
            [0.52, 0.61, 0.71],
            [1, 1, 1],
            [1.92, 2.48, 2.96],
            [1.64, 2.24, 2.75],
        ],
        [
            [0.26, 0.34, 0.47],
            [0.25, 0.30, 0.38],
            [0.34, 0.45, 0.69],
            [0.34, 0.40, 0.52],
            [1, 1, 1],
            [0.95, 1.12, 1.25]
        ],
        [
            [0.65, 0.78, 0.95],
            [0.31, 0.37, 0.47],
            [0.39, 0.47, 0.60],
            [0.36, 0.45, 0.61],
            [0.80, 0.90, 1.06],
            [1, 1, 1]
        ]
    ])
    calculated_weights = f_ahp(comparison_matrix)
    
    reference_weights = np.array([
        [0.147, 0.208, 0.286],
        [0.186, 0.261, 0.358],
        [0.133, 0.187, 0.269],
        [0.124, 0.169, 0.227],
        [0.057, 0.080, 0.119],
        [0.070, 0.094, 0.136]
    ])
    assert (np.abs((np.round(calculated_weights, 3) - reference_weights)) < 0.01).all()

def test_fBWM_weights():
    """
        Test verifying correctness of the fuzzy BWM methods.
        Reference value: Guo, S., & Zhao, H. (2017). Fuzzy best-worst multi-criteria decision-making method and its applications. Knowledge-Based Systems, 121, 23-31.
    """

    f_bwm = fBWM()
    best_to_others = np.array([
        [7/2, 4, 9/2],
        [2/3, 1, 3/2],
        [1, 1, 1]
    ])

    others_to_worst = np.array([
        [1, 1, 1],
        [3/2, 2, 5/2],
        [7/2, 4, 9/2]
    ])

    calculated_weights = f_bwm(
        best_to_others,
        others_to_worst,
        best_idx=2,
        worst_idx=0
    )
    reference_weights = np.array([
        [0.1341, 0.1449, 0.1449],
        [0.2823, 0.3550, 0.3952],
        [0.4423, 0.5146, 0.5431]
    ])
    assert (np.abs((np.round(calculated_weights, 3) - reference_weights)) < 0.01).all()

def test_fFUCOM_weights():
    """
        Test verifying correctness of the fuzzy FUCOM methods.
        Reference value: Pamucar, D., & Ecer, F. (2020). Prioritizing the weights of the evaluation criteria under fuzziness: the fuzzy full consistency method - FUCOM-F. Facta Universitatis, Series: Mechanical Engineering, 18(3), 419-437.
    """

    f_fucom = fFUCOM()
    order = [1, 2, 3]
    significance = ['EI', 'WI', 'FI'] 
    calculated_weights = f_fucom(criteria_order=order, significance=significance)
    
    reference_weights = np.array([
        [0.261, 0.3891, 0.5831],
        [0.3881, 0.3881, 0.3881],
        [0.1038, 0.1945, 0.3891]
    ])
    assert (np.abs((np.round(calculated_weights, 3) - reference_weights)) < 0.01).all()

def test_fLMAW_weights():
    """
        Test verifying correctness of the fuzzy LMAW methods.
        Reference value: Božanić, D., Pamučar, D., Milić, A., Marinković, D., & Komazec, N. (2022). Modification of the logarithm methodology of additive weights (LMAW) by a triangular fuzzy number and its application in multi-criteria decision making. Axioms, 11(3), 89.
    """

    f_lmaw = fLMAW()
    P1 = ("AH", "L", "VL", "E", "VL", "VL")
    P2 = ("AH", "ML", "AL", "H", "AL", "VL")
    P3 = ("AH", "ML", "AL", "MH", "VL", "L")
    P4 = ("AH", "E", "AL", "VH", "AL", "AL")

    expert_inputs = [P1, P2, P3, P4]
    calculated_weights = f_lmaw(expert_inputs)
    
    reference_weights = np.array([
        [0.2343, 0.2659, 0.3032],
        [0.1448, 0.1839, 0.2345],
        [0.0739, 0.0898, 0.1085],
        [0.1971, 0.2307, 0.2801],
        [0.0739, 0.1008, 0.1291],
        [0.0829, 0.1198, 0.1593]
    ])
    assert (np.abs((np.round(calculated_weights, 3) - reference_weights)) < 0.01).all()

def test_fRANCOM_weights():
    """
        Test verifying correctness of the fuzzy RANCOM methods.
        Reference value: Więckowski, J., Kizielewicz, B., & Sałabun, W. (2025). Fuzzy RANCOM: a novel approach for modeling uncertainty in decision-making processes. Information sciences, 694, 121716.
    """

    f_rancom = fRANCOM()
    ranking = np.array([2, 1, 4, 3])
    calculated_weights = f_rancom(ranking)
    
    reference_weights = np.array([
        [0.1250, 0.3125, 0.4375],
        [0.1875, 0.4375, 0.5000],
        [0.0000, 0.0625, 0.3125],
        [0.0625, 0.1875, 0.3750]
    ])
    assert (np.abs((np.round(calculated_weights, 3) - reference_weights)) < 0.01).all()

def test_fSWARA_weights():
    """
        Test verifying correctness of the fuzzy SWARA methods.
        Reference value: Mehdiabadi, A., Sadeghi, A., Karbassi Yazdi, A., & Tan, Y. (2025). Sustainability Service Chain Capabilities in the Oil and Gas Industry: A Fuzzy Hybrid Approach SWARA-MABAC. Spectrum of Operational Research, 2(1), 114-134.    
    """

    f_swara = fSWARA()
    ranking = np.array([1, 2, 3, 4, 5, 6, 7])

    comparative_importance = [
        [1.32, 0.666, 0.55],   
        [0.274, 0.75, 0.322],  
        [0.474, 1.025, 1.635], 
        [1.0, 1.0, 1.0],       
        [0.286, 0.353, 0.258], 
        [0.243, 0.456, 0.147]  
    ]

    calculated_weights = f_swara(ranking=ranking, comparative_importance=comparative_importance)
    reference_weights = np.array([
        [0.440, 0.435, 0.393],
        [0.189, 0.261, 0.253],
        [0.148, 0.148, 0.191],
        [0.100, 0.073, 0.072],
        [0.050, 0.036, 0.036],
        [0.038, 0.026, 0.028],
        [0.030, 0.018, 0.024]
    ])
    assert (np.abs((np.round(calculated_weights, 3) - reference_weights)) < 0.01).all()
