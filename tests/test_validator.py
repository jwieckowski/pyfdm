# Copyright (c) 2026 Jakub Więckowski

import numpy as np
import pytest

from pyfdm.validator import Validator

# validate_matrix_shape
def test_validate_matrix_shape_correct():
    """
        Test verifying that a well-formed (m, n, 3) decision matrix passes
        shape validation without raising.
    """
    matrix = np.zeros((3, 4, 3))
    Validator.validate_matrix_shape(matrix)
def test_validate_matrix_shape_not_ndarray():
    """
        Test verifying that a non-ndarray decision matrix raises TypeError.
    """
    with pytest.raises(TypeError):
        Validator.validate_matrix_shape([[1, 2, 3]])
def test_validate_matrix_shape_wrong_ndim():
    """
        Test verifying that a 2-D matrix (missing the TFN dimension) raises
        ValueError.
    """
    matrix = np.zeros((3, 4))
    with pytest.raises(ValueError):
        Validator.validate_matrix_shape(matrix)
def test_validate_matrix_shape_wrong_last_dim():
    """
        Test verifying that a matrix whose last dimension is not 3 (TFN
        components) raises ValueError.
    """
    matrix = np.zeros((3, 4, 2))
    with pytest.raises(ValueError):
        Validator.validate_matrix_shape(matrix)
def test_validate_matrix_shape_too_few_alternatives():
    """
        Test verifying that a matrix with fewer than 2 alternatives raises
        ValueError.
    """
    matrix = np.zeros((1, 4, 3))
    with pytest.raises(ValueError):
        Validator.validate_matrix_shape(matrix)
def test_validate_matrix_shape_too_few_criteria():
    """
        Test verifying that a matrix with fewer than 1 criterion raises
        ValueError.
    """
    matrix = np.zeros((3, 0, 3))
    with pytest.raises(ValueError):
        Validator.validate_matrix_shape(matrix)
# validate_tfn_values
def test_validate_tfn_values_correct():
    """
        Test verifying that a decision matrix with valid TFNs (l <= m <= u,
        no NaN/Inf) passes validation without raising.
    """
    matrix = np.array([[[1, 2, 3], [2, 3, 4]], [[1, 1, 2], [0, 1, 1]]])
    Validator.validate_tfn_values(matrix)

def test_validate_tfn_values_nan():
    """
        Test verifying that a decision matrix containing NaN values raises
        ValueError.
    """
    matrix = np.array([[[1, 2, 3], [2, np.nan, 4]]])
    with pytest.raises(ValueError):
        Validator.validate_tfn_values(matrix)

def test_validate_tfn_values_inf():
    """
        Test verifying that a decision matrix containing Inf values raises
        ValueError.
    """
    matrix = np.array([[[1, 2, 3], [2, np.inf, 4]]])
    with pytest.raises(ValueError):
        Validator.validate_tfn_values(matrix)

def test_validate_tfn_values_not_monotonic():
    """
        Test verifying that a TFN violating l <= m <= u raises ValueError.
    """
    matrix = np.array([[[3, 2, 1]]])
    with pytest.raises(ValueError):
        Validator.validate_tfn_values(matrix)

# validate_tfn
def test_validate_tfn_correct():
    """
        Test verifying that a well-formed 3-element TFN passes validation
        without raising.
    """
    Validator.validate_tfn([1, 2, 3], 'ref_point')

def test_validate_tfn_wrong_shape():
    """
        Test verifying that a TFN with an incorrect number of components
        raises ValueError.
    """
    with pytest.raises(ValueError):
        Validator.validate_tfn([1, 2], 'ref_point')

def test_validate_tfn_non_numeric():
    """
        Test verifying that a TFN containing non-numeric data raises
        ValueError.
    """
    with pytest.raises(ValueError):
        Validator.validate_tfn(['a', 'b', 'c'], 'ref_point')

# validate_weights
def test_validate_weights_crisp_correct():
    """
        Test verifying that valid crisp weights (non-negative, sum to 1)
        pass validation without raising.
    """
    weights = np.array([0.2, 0.3, 0.5])
    Validator.validate_weights(weights)

def test_validate_weights_not_ndarray():
    """
        Test verifying that non-ndarray weights raise TypeError.
    """
    with pytest.raises(TypeError):
        Validator.validate_weights([0.5, 0.5])

def test_validate_weights_crisp_negative():
    """
        Test verifying that negative crisp weights raise ValueError.
    """
    weights = np.array([-0.1, 0.6, 0.5])
    with pytest.raises(ValueError):
        Validator.validate_weights(weights)

def test_validate_weights_crisp_not_summing_to_one():
    """
        Test verifying that crisp weights not summing to 1 raise
        ValueError.
    """
    weights = np.array([0.2, 0.2, 0.2])
    with pytest.raises(ValueError):
        Validator.validate_weights(weights)

def test_validate_weights_crisp_required_but_fuzzy_given():
    """
        Test verifying that fuzzy (2-D) weights raise ValueError when
        crisp_required=True.
    """
    weights = np.array([[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])
    with pytest.raises(ValueError):
        Validator.validate_weights(weights, crisp_required=True)

def test_validate_weights_fuzzy_correct():
    """
        Test verifying that valid fuzzy (TFN) weights pass validation
        without raising.
    """
    weights = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    Validator.validate_weights(weights)

def test_validate_weights_fuzzy_wrong_last_dim():
    """
        Test verifying that fuzzy weights without exactly 3 TFN components
        raise ValueError.
    """
    weights = np.array([[0.1, 0.2], [0.2, 0.3]])
    with pytest.raises(ValueError):
        Validator.validate_weights(weights)

def test_validate_weights_fuzzy_negative():
    """
        Test verifying that negative fuzzy weight components raise
        ValueError.
    """
    weights = np.array([[-0.1, 0.2, 0.3]])
    with pytest.raises(ValueError):
        Validator.validate_weights(weights)

def test_validate_weights_fuzzy_not_monotonic():
    """
        Test verifying that a fuzzy weight violating l <= m <= u raises
        ValueError.
    """
    weights = np.array([[0.3, 0.2, 0.1]])
    with pytest.raises(ValueError):
        Validator.validate_weights(weights)

def test_validate_weights_wrong_ndim():
    """
        Test verifying that weights with more than 2 dimensions raise
        ValueError.
    """
    weights = np.zeros((2, 2, 3, 1))
    with pytest.raises(ValueError):
        Validator.validate_weights(weights)

# validate_types
def test_validate_types_correct():
    """
        Test verifying that a valid criteria types vector (1 / -1) passes
        validation without raising.
    """
    types = np.array([1, -1, 1])
    Validator.validate_types(types)

def test_validate_types_none_skipped():
    """
        Test verifying that passing None for types skips validation without
        raising.
    """
    Validator.validate_types(None)

def test_validate_types_not_ndarray():
    """
        Test verifying that non-ndarray types raise TypeError.
    """
    with pytest.raises(TypeError):
        Validator.validate_types([1, -1])

def test_validate_types_invalid_value():
    """
        Test verifying that a types vector with values other than 1 or -1
        raises ValueError.
    """
    types = np.array([1, 0, -1])
    with pytest.raises(ValueError):
        Validator.validate_types(types)

def test_validate_types_different_types_required_but_same():
    """
        Test verifying that requiring different_types=True with a
        single-type vector raises ValueError.
    """
    types = np.array([1, 1, 1])
    with pytest.raises(ValueError):
        Validator.validate_types(types, different_types=True)

def test_validate_types_different_types_satisfied():
    """
        Test verifying that different_types=True passes when both profit
        and cost criteria are present.
    """
    types = np.array([1, -1, 1])
    Validator.validate_types(types, different_types=True)

# validate_input
def test_validate_input_correct():
    """
        Test verifying that consistent matrix, weights and types
        dimensions pass validation without raising.
    """
    matrix = np.zeros((3, 2, 3))
    weights = np.array([0.5, 0.5])
    types = np.array([1, -1])
    Validator.validate_input(matrix, weights, types)

def test_validate_input_weights_mismatch():
    """
        Test verifying that a mismatch between the number of criteria and
        the number of weights raises ValueError.
    """
    matrix = np.zeros((3, 2, 3))
    weights = np.array([0.5, 0.3, 0.2])
    with pytest.raises(ValueError):
        Validator.validate_input(matrix, weights=weights)

def test_validate_input_types_mismatch():
    """
        Test verifying that a mismatch between the number of criteria and
        the number of types raises ValueError.
    """
    matrix = np.zeros((3, 2, 3))
    types = np.array([1, -1, 1])
    with pytest.raises(ValueError):
        Validator.validate_input(matrix, types=types)

# fuzzy_validation
def test_fuzzy_validation_correct():
    """
        Test verifying that a fully consistent fuzzy MCDA problem (matrix,
        weights, types) passes the combined validation pipeline without
        raising.
    """
    matrix = np.array([
        [[1, 2, 3], [2, 3, 4]],
        [[2, 3, 4], [1, 2, 3]],
    ])
    weights = np.array([0.5, 0.5])
    types = np.array([1, -1])
    Validator.fuzzy_validation(matrix, weights, types, crisp_required=True, different_types=True)

def test_fuzzy_validation_propagates_matrix_error():
    """
        Test verifying that an invalid decision matrix causes
        fuzzy_validation to raise ValueError (propagated from
        validate_matrix_shape).
    """
    matrix = np.zeros((1, 2, 3))
    weights = np.array([0.5, 0.5])
    types = np.array([1, -1])
    with pytest.raises(ValueError):
        Validator.fuzzy_validation(matrix, weights, types, crisp_required=True, different_types=False)

# validate_comparison_matrix
def test_validate_comparison_matrix_fuzzy_correct():
    """
        Test verifying that a well-formed square fuzzy (n, n, 3) comparison
        matrix passes validation without raising.
    """
    matrix = np.zeros((3, 3, 3))
    Validator.validate_comparison_matrix(matrix, dim=3)

def test_validate_comparison_matrix_crisp_correct():
    """
        Test verifying that a well-formed square crisp (n, n) comparison
        matrix passes validation without raising.
    """
    matrix = np.zeros((3, 3))
    Validator.validate_comparison_matrix(matrix, dim=2)

def test_validate_comparison_matrix_not_ndarray():
    """
        Test verifying that a non-ndarray comparison matrix raises
        TypeError.
    """
    with pytest.raises(TypeError):
        Validator.validate_comparison_matrix([[1, 2], [3, 4]])

def test_validate_comparison_matrix_invalid_dim_argument():
    """
        Test verifying that an unsupported 'dim' argument raises
        ValueError.
    """
    matrix = np.zeros((3, 3))
    with pytest.raises(ValueError):
        Validator.validate_comparison_matrix(matrix, dim=4)

def test_validate_comparison_matrix_wrong_ndim():
    """
        Test verifying that a matrix whose number of dimensions doesn't
        match 'dim' raises ValueError.
    """
    matrix = np.zeros((3, 3))
    with pytest.raises(ValueError):
        Validator.validate_comparison_matrix(matrix, dim=3)

def test_validate_comparison_matrix_not_square():
    """
        Test verifying that a non-square comparison matrix raises
        ValueError.
    """
    matrix = np.zeros((2, 3))
    with pytest.raises(ValueError):
        Validator.validate_comparison_matrix(matrix, dim=2)

def test_validate_comparison_matrix_fuzzy_wrong_last_dim():
    """
        Test verifying that a fuzzy comparison matrix without exactly 3 TFN
        components raises ValueError.
    """
    matrix = np.zeros((3, 3, 2))
    with pytest.raises(ValueError):
        Validator.validate_comparison_matrix(matrix, dim=3)

# validate_param_range
def test_validate_param_range_correct():
    """
        Test verifying that a numeric value within the given interval
        passes validation without raising.
    """
    Validator.validate_param_range(0.5, 0.0, 1.0, 'alpha')

def test_validate_param_range_non_numeric():
    """
        Test verifying that a non-numeric parameter raises TypeError.
    """
    with pytest.raises(TypeError):
        Validator.validate_param_range('0.5', 0.0, 1.0, 'alpha')

def test_validate_param_range_out_of_bounds():
    """
        Test verifying that a value outside the given interval raises
        ValueError.
    """
    with pytest.raises(ValueError):
        Validator.validate_param_range(1.5, 0.0, 1.0, 'alpha')

def test_validate_param_range_boundary_values():
    """
        Test verifying that values exactly at the interval boundaries pass
        validation without raising.
    """
    Validator.validate_param_range(0.0, 0.0, 1.0, 'alpha')
    Validator.validate_param_range(1.0, 0.0, 1.0, 'alpha')

# validate_vectors
def test_validate_vectors_correct():
    """
        Test verifying that two equal-length numeric vectors are validated
        and converted to float arrays correctly.
    """
    x = [1, 2, 3]
    y = [4, 5, 6]
    x_out, y_out = Validator.validate_vectors(x, y)
    assert np.array_equal(x_out, np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(y_out, np.array([4.0, 5.0, 6.0]))

def test_validate_vectors_non_numeric():
    """
        Test verifying that non-numeric vector entries raise ValueError.
    """
    with pytest.raises(ValueError):
        Validator.validate_vectors(['a', 'b'], [1, 2])

def test_validate_vectors_wrong_ndim():
    """
        Test verifying that a multi-dimensional vector input raises
        ValueError.
    """
    with pytest.raises(ValueError):
        Validator.validate_vectors([[1, 2], [3, 4]], [1, 2])

def test_validate_vectors_length_mismatch():
    """
        Test verifying that two vectors of different lengths raise
        ValueError.
    """
    with pytest.raises(ValueError):
        Validator.validate_vectors([1, 2, 3], [1, 2])

def test_validate_vectors_empty():
    """
        Test verifying that empty input vectors raise ValueError.
    """
    with pytest.raises(ValueError):
        Validator.validate_vectors([], [])

# validate_ervd_input
def test_validate_ervd_input_correct():
    """
        Test verifying that a well-formed (n, 3) reference point passes
        validation and is returned as a float ndarray.
    """
    ref_point = [[1, 2, 3], [2, 3, 4]]
    result = Validator.validate_ervd_input(ref_point, n=2)
    assert result.shape == (2, 3)

def test_validate_ervd_input_none():
    """
        Test verifying that a None reference point is accepted and returns
        None without raising.
    """
    result = Validator.validate_ervd_input(None, n=2)
    assert result is None

def test_validate_ervd_input_wrong_shape():
    """
        Test verifying that a reference point without 3 TFN components per
        criterion raises ValueError.
    """
    ref_point = [[1, 2], [2, 3]]
    with pytest.raises(ValueError):
        Validator.validate_ervd_input(ref_point, n=2)

def test_validate_ervd_input_criteria_mismatch():
    """
        Test verifying that a reference point whose number of criteria
        doesn't match the decision matrix raises ValueError.
    """
    ref_point = [[1, 2, 3], [2, 3, 4]]
    with pytest.raises(ValueError):
        Validator.validate_ervd_input(ref_point, n=3)

def test_validate_ervd_input_non_numeric():
    """
        Test verifying that a non-numeric reference point raises
        ValueError.
    """
    ref_point = [['a', 'b', 'c']]
    with pytest.raises(ValueError):
        Validator.validate_ervd_input(ref_point, n=1)

# validate_rim_input
def test_validate_rim_input_correct():
    """
        Test verifying that consistent, well-formed RIM bounds pass
        validation without raising.
    """
    matrix = np.zeros((3, 2, 3))
    lower_bound = [[0, 0, 0], [0, 0, 0]]
    upper_bound = [[1, 1, 1], [1, 1, 1]]
    Validator.validate_rim_input(matrix, lower_bound=lower_bound, upper_bound=upper_bound)

def test_validate_rim_input_all_none():
    """
        Test verifying that all bounds being None is accepted without
        raising.
    """
    matrix = np.zeros((3, 2, 3))
    Validator.validate_rim_input(matrix)

def test_validate_rim_input_criteria_mismatch():
    """
        Test verifying that a bounds array with an inconsistent number of
        criteria raises ValueError.
    """
    matrix = np.zeros((3, 2, 3))
    lower_bound = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    with pytest.raises(ValueError):
        Validator.validate_rim_input(matrix, lower_bound=lower_bound)

def test_validate_rim_input_non_numeric():
    """
        Test verifying that a non-numeric bounds array raises ValueError.
    """
    matrix = np.zeros((3, 2, 3))
    lower_reference = [['a', 'b', 'c'], ['d', 'e', 'f']]
    with pytest.raises(ValueError):
        Validator.validate_rim_input(matrix, lower_reference=lower_reference)

# validate_rafsi_input
def test_validate_rafsi_input_correct():
    """
        Test verifying that a valid RAFSI configuration (proper interval,
        matching ideal/anti-ideal lengths) passes validation without
        raising.
    """
    matrix = np.zeros((3, 2, 3))
    ideal = [10, 10]
    anti_ideal = [0, 0]
    Validator.validate_rafsi_input(matrix, lower_bound=1.0, upper_bound=6.0, ideal=ideal, anti_ideal=anti_ideal)

def test_validate_rafsi_input_none_bounds():
    """
        Test verifying that ideal/anti_ideal being None is accepted
        without raising.
    """
    matrix = np.zeros((3, 2, 3))
    Validator.validate_rafsi_input(matrix, lower_bound=1.0, upper_bound=6.0, ideal=None, anti_ideal=None)

def test_validate_rafsi_input_invalid_interval():
    """
        Test verifying that lower_bound >= upper_bound raises ValueError.
    """
    matrix = np.zeros((3, 2, 3))
    with pytest.raises(ValueError):
        Validator.validate_rafsi_input(matrix, lower_bound=6.0, upper_bound=1.0, ideal=None, anti_ideal=None)

def test_validate_rafsi_input_ideal_length_mismatch():
    """
        Test verifying that an 'ideal' vector with a length inconsistent
        with the number of criteria raises ValueError.
    """
    matrix = np.zeros((3, 2, 3))
    ideal = [10, 10, 10]
    with pytest.raises(ValueError):
        Validator.validate_rafsi_input(matrix, lower_bound=1.0, upper_bound=6.0, ideal=ideal, anti_ideal=None)

# validate_spotis_input
def test_validate_spotis_input_correct():
    """
        Test verifying that well-formed SPOTIS bounds, matching the number
        of criteria, pass validation without raising.
    """
    matrix = np.zeros((3, 2, 3))
    bounds = [[0, 10], [0, 20]]
    Validator.validate_spotis_input(matrix, bounds)

def test_validate_spotis_input_none():
    """
        Test verifying that bounds being None is accepted without raising.
    """
    matrix = np.zeros((3, 2, 3))
    Validator.validate_spotis_input(matrix, None)

def test_validate_spotis_input_wrong_shape():
    """
        Test verifying that bounds without exactly 2 columns (min, max)
        raise ValueError.
    """
    matrix = np.zeros((3, 2, 3))
    bounds = [[0, 10, 20], [0, 20, 30]]
    with pytest.raises(ValueError):
        Validator.validate_spotis_input(matrix, bounds)

def test_validate_spotis_input_criteria_mismatch():
    """
        Test verifying that a bounds array with an inconsistent number of
        rows (criteria) raises ValueError.
    """
    matrix = np.zeros((3, 2, 3))
    bounds = [[0, 10], [0, 20], [0, 30]]
    with pytest.raises(ValueError):
        Validator.validate_spotis_input(matrix, bounds)

# validate_lmaw_input
def test_validate_lmaw_input_tfn_single_expert_correct():
    """
        Test verifying that a valid single-expert TFN input, shape
        (n_criteria, 3), passes validation without raising.
    """
    expert_decisions = np.array([[1, 2, 3], [2, 3, 4]])
    Validator.validate_lmaw_input(expert_decisions)

def test_validate_lmaw_input_tfn_multi_expert_correct():
    """
        Test verifying that a valid multi-expert TFN input, shape
        (n_experts, n_criteria, 3), passes validation without raising.
    """
    expert_decisions = np.array([[[1, 2, 3], [2, 3, 4]], [[1, 1, 2], [2, 2, 3]]])
    Validator.validate_lmaw_input(expert_decisions)

def test_validate_lmaw_input_linguistic_correct():
    """
        Test verifying that valid linguistic-term input, resolvable via the
        provided linguistic scale, passes validation without raising.
    """
    scale = {'L': (1, 1, 1), 'M': (2, 2, 2), 'H': (3, 3, 3)}
    expert_decisions = ['L', 'M', 'H']
    Validator.validate_lmaw_input(expert_decisions, linguistic_scale=scale)

def test_validate_lmaw_input_linguistic_missing_scale():
    """
        Test verifying that linguistic-term input without a provided
        linguistic_scale raises ValueError.
    """
    expert_decisions = ['L', 'M', 'H']
    with pytest.raises(ValueError):
        Validator.validate_lmaw_input(expert_decisions, linguistic_scale=None)

def test_validate_lmaw_input_unknown_linguistic_term():
    """
        Test verifying that a linguistic term absent from the provided
        scale raises ValueError.
    """
    scale = {'L': (1, 1, 1), 'M': (2, 2, 2)}
    expert_decisions = ['L', 'X']
    with pytest.raises(ValueError):
        Validator.validate_lmaw_input(expert_decisions, linguistic_scale=scale)

# validate_bwm_input
def test_validate_bwm_input_correct():
    """
        Test verifying that consistent, well-formed BWM comparison vectors
        and valid best/worst indices pass validation without raising.
    """
    best_to_others = np.array([[1, 1, 1], [2, 3, 4], [3, 4, 5]])
    others_to_worst = np.array([[3, 4, 5], [2, 3, 4], [1, 1, 1]])
    Validator.validate_bwm_input(best_to_others, others_to_worst, best_idx=0, worst_idx=2)

def test_validate_bwm_input_wrong_shape():
    """
        Test verifying that a best_to_others vector without shape (n, 3)
        raises ValueError.
    """
    best_to_others = np.array([[1, 1], [2, 3]])
    others_to_worst = np.array([[1, 1, 1], [2, 3, 4]])
    with pytest.raises(ValueError):
        Validator.validate_bwm_input(best_to_others, others_to_worst, best_idx=0, worst_idx=1)

def test_validate_bwm_input_shape_mismatch():
    """
        Test verifying that others_to_worst with a shape different from
        best_to_others raises ValueError.
    """
    best_to_others = np.array([[1, 1, 1], [2, 3, 4]])
    others_to_worst = np.array([[1, 1, 1], [2, 3, 4], [3, 4, 5]])
    with pytest.raises(ValueError):
        Validator.validate_bwm_input(best_to_others, others_to_worst, best_idx=0, worst_idx=1)

def test_validate_bwm_input_invalid_best_idx():
    """
        Test verifying that a best_idx outside the valid criteria index
        range raises ValueError.
    """
    best_to_others = np.array([[1, 1, 1], [2, 3, 4]])
    others_to_worst = np.array([[2, 3, 4], [1, 1, 1]])
    with pytest.raises(ValueError):
        Validator.validate_bwm_input(best_to_others, others_to_worst, best_idx=5, worst_idx=1)

def test_validate_bwm_input_invalid_worst_idx():
    """
        Test verifying that a worst_idx outside the valid criteria index
        range raises ValueError.
    """
    best_to_others = np.array([[1, 1, 1], [2, 3, 4]])
    others_to_worst = np.array([[2, 3, 4], [1, 1, 1]])
    with pytest.raises(ValueError):
        Validator.validate_bwm_input(best_to_others, others_to_worst, best_idx=0, worst_idx=-1)