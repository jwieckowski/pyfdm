# Copyright (c) 2022-2023 Jakub Więckowski

import numpy as np
import pyfdm.methods.utils.defuzzifications as dfs


def test_mean_defuzzification():
    """
        Test veryfing correctness of the mean defuzzification formula.
        Reference value: Yılmaz, M., & Atan, T. (2021). Hospital site selection using fuzzy EDAS method: case study application for districts of Istanbul. Journal of Intelligent & Fuzzy Systems, (Preprint), 1-12.
    """
    x = np.array([-2.551, 0.198, 2.990])
    calculated_value = dfs.mean_defuzzification(x)
    reference_value = 0.212

    assert np.round(calculated_value, 3) == reference_value


def test_mean_area_defuzzification():
    """
        Test veryfing correctness of the weighted mean defuzzification formula.
        Reference value: Opricovic, S. (2007). A fuzzy compromise solution for multicriteria problems. International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, 15(03), 363-380.
    """
    x = np.array([3.955, 5.919, 14.372])
    calculated_value = dfs.mean_area_defuzzification(x)
    reference_value = 7.541

    assert np.round(calculated_value, 3) == reference_value


def test_graded_mean_average_defuzzification():
    """
        Test veryfing correctness of the graded mean average defuzzification formula.
        Reference value: Zindani, D., Maity, S. R., & Bhowmik, S. (2019). Fuzzy-EDAS (evaluation based on distance from average solution) for material selection problems. In Advances in Computational Methods in Manufacturing (pp. 755-771). Springer, Singapore.
    """
    x = np.array([0.58, 0.77, 0.91])
    calculated_value = dfs.graded_mean_average_defuzzification(x)
    reference_value = 0.76

    assert np.round(calculated_value, 2) == reference_value

def test_weighted_mean_defuzzification():
    """
        Test veryfing correctness of the weighted mean defuzzification formula.
        Reference value: Opricovic, S. (2007). A fuzzy compromise solution for multicriteria problems. International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, 15(03), 363-380.
    """
    x = np.array([3.955, 5.919, 14.372])
    calculated_value = dfs.weighted_mean_defuzzification(x)
    reference_value = 7.541

    assert np.round(calculated_value, 3) == reference_value

def test_bisector_defuzzification():
    """
        Test veryfing correctness of the bisector defuzzification formula.
        Formula: Berkachy, R., & Donzé, L. (2016). Linguistic questionnaire evaluation: an application of the signed distance defuzzification method on different fuzzy numbers. The impact on the skewness of the output distributions. International Journal of Fuzzy Systems and Advanced Applications, 3, 12-19.
        Reference value: Self-calculated empirical verification 
    """
    x = np.array([3.955, 5.919, 14.372])
    calculated_value = dfs.bisector_defuzzification(x)
    reference_value = (x[0] + x[2]) / 2

    assert np.round(calculated_value, 3) == np.round(reference_value, 3)

def test_height_defuzzification():
    """
        Test veryfing correctness of the height defuzzification formula.
        Formula: Berkachy, R., & Donzé, L. (2016). Linguistic questionnaire evaluation: an application of the signed distance defuzzification method on different fuzzy numbers. The impact on the skewness of the output distributions. International Journal of Fuzzy Systems and Advanced Applications, 3, 12-19.
        Reference value: Self-calculated empirical verification 
    """
    x = np.array([3.955, 5.919, 14.372])
    calculated_value = dfs.height_defuzzification(x)
    reference_value = x[1]

    assert np.round(calculated_value, 3) == np.round(reference_value, 3)

def test_lom_defuzzification():
    """
        Test veryfing correctness of the lom defuzzification formula.
        Formula: Berkachy, R., & Donzé, L. (2016). Linguistic questionnaire evaluation: an application of the signed distance defuzzification method on different fuzzy numbers. The impact on the skewness of the output distributions. International Journal of Fuzzy Systems and Advanced Applications, 3, 12-19.
        Reference value: Self-calculated empirical verification 
    """
    x = np.array([3.955, 5.919, 14.372])
    calculated_value = dfs.lom_defuzzification(x)
    reference_value = x[2]

    assert np.round(calculated_value, 3) == np.round(reference_value, 3) 

def test_som_defuzzification():
    """
        Test veryfing correctness of the som defuzzification formula.
        Formula: Berkachy, R., & Donzé, L. (2016). Linguistic questionnaire evaluation: an application of the signed distance defuzzification method on different fuzzy numbers. The impact on the skewness of the output distributions. International Journal of Fuzzy Systems and Advanced Applications, 3, 12-19.
        Reference value: Self-calculated empirical verification 
    """
    x = np.array([3.955, 5.919, 14.372])
    calculated_value = dfs.som_defuzzification(x)
    reference_value = x[0]

    assert np.round(calculated_value, 3) ==  np.round(reference_value, 3)
