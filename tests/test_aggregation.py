# Copyright (c) 2026 Jakub Więckowski

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyfdm.group import aggregate

@pytest.fixture
def small_matrix():
    """3 alternatives, 3 criteria, all valid TFNs."""
    return np.array([
        [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
        [(2, 3, 4), (5, 6, 7), (1, 2, 3)],
        [(3, 4, 5), (2, 3, 4), (4, 5, 6)],
    ], dtype=float)


class TestGroupAggregation:

    @pytest.fixture
    def three_matrices(self, small_matrix):
        return [small_matrix, small_matrix * 1.1, small_matrix * 0.9]

    def test_geometric_mean_shape(self, three_matrices, small_matrix):
        agg = aggregate(three_matrices, method='geometric')
        assert agg.shape == small_matrix.shape

    def test_arithmetic_mean_shape(self, three_matrices, small_matrix):
        agg = aggregate(three_matrices, method='arithmetic')
        assert agg.shape == small_matrix.shape

    def test_arithmetic_mean_of_identical(self, small_matrix):
        agg = aggregate([small_matrix, small_matrix], method='arithmetic')
        assert np.allclose(agg, small_matrix)

    def test_weighted_average_shape(self, three_matrices, small_matrix):
        agg = aggregate(three_matrices, method='weighted', weights=[0.5, 0.3, 0.2])
        assert agg.shape == small_matrix.shape

    def test_weighted_average_normalizes_weights(self, three_matrices):
        # Unnormalized weights [5, 3, 2] should give same result as [0.5, 0.3, 0.2]
        agg1 = aggregate(three_matrices, method='weighted', weights=[0.5, 0.3, 0.2])
        agg2 = aggregate(three_matrices, method='weighted', weights=[5.0, 3.0, 2.0])
        assert np.allclose(agg1, agg2)

    def test_owa_shape(self, three_matrices, small_matrix):
        agg = aggregate(three_matrices, method='owa', owa_weights=[0.5, 0.3, 0.2])
        assert agg.shape == small_matrix.shape

    def test_invalid_method_raises(self, three_matrices):
        with pytest.raises(ValueError, match='Unknown aggregation'):
            aggregate(three_matrices, method='invalid')

    def test_empty_matrices_raises(self):
        with pytest.raises(ValueError, match='empty'):
            aggregate([])

    def test_weighted_wrong_length_raises(self, three_matrices):
        with pytest.raises(ValueError, match='Length of weights'):
            aggregate(three_matrices, method='weighted', weights=[0.5, 0.5])

    def test_geometric_mean_all_positive(self, three_matrices):
        agg = aggregate(three_matrices, method='geometric')
        assert np.all(agg > 0)
