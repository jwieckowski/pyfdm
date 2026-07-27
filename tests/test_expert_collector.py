# Copyright (c) 2026 Jakub Więckowski

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyfdm.expert import ExpertCollector
from pyfdm.expert.scales import SCALE_1_9

class TestExpertCollector:

    @pytest.fixture
    def collector(self):
        c = ExpertCollector(n_alternatives=3, n_criteria=3, scale=SCALE_1_9)
        c.add_expert_matrix(0, [[9, 3, 7], [5, 9, 3], [3, 5, 9]])
        c.add_expert_matrix(1, [[7, 5, 9], [9, 3, 5], [5, 7, 3]])
        return c

    def test_n_experts(self, collector):
        assert collector.n_experts == 2

    def test_expert_ids(self, collector):
        assert collector.expert_ids == [0, 1]

    def test_tfn_matrix_shape(self, collector):
        m = collector.get_tfn_matrix(0)
        assert m.shape == (3, 3, 3)

    def test_tfn_values_in_range(self, collector):
        m = collector.get_tfn_matrix(0)
        assert np.all(m >= 0) and np.all(m <= 9)

    def test_get_tfn_matrices_length(self, collector):
        mats = collector.get_tfn_matrices()
        assert len(mats) == 2

    def test_invalid_rating_raises(self):
        c = ExpertCollector(n_alternatives=2, n_criteria=2, scale=SCALE_1_9)
        with pytest.raises(ValueError, match='not in scale'):
            c.add_expert_matrix(0, [[10, 3], [5, 9]])

    def test_wrong_shape_raises(self):
        c = ExpertCollector(n_alternatives=2, n_criteria=2, scale=SCALE_1_9)
        with pytest.raises(ValueError, match='shape'):
            c.add_expert_matrix(0, [[9, 3, 7], [5, 9, 3]])

    def test_missing_expert_raises(self, collector):
        with pytest.raises(KeyError):
            collector.get_tfn_matrix(99)

    def test_repr(self, collector):
        r = repr(collector)
        assert 'ExpertCollector' in r
        assert 'n_experts=2' in r
