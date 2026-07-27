# Copyright (c) 2026 Jakub Więckowski

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyfdm.utils.ranking import rank_alternatives
from pyfdm.helpers import rank as legacy_rank


# Basic behaviour
class TestBasicRanking:

    def test_simple_descending_no_ties(self):
        scores = [0.2, 0.8, 0.5]
        r = rank_alternatives(scores)
        # 0.8 is best (rank 1), 0.5 is rank 2, 0.2 is rank 3
        assert r.tolist() == [3.0, 1.0, 2.0]

    def test_simple_ascending_no_ties(self):
        scores = [0.2, 0.8, 0.5]
        r = rank_alternatives(scores, descending=False)
        # lower is better: 0.2 -> rank 1, 0.5 -> rank 2, 0.8 -> rank 3
        assert r.tolist() == [1.0, 3.0, 2.0]

    def test_single_alternative(self):
        r = rank_alternatives([0.5])
        assert r.tolist() == [1.0]

    def test_accepts_list(self):
        r = rank_alternatives([3, 1, 2])
        assert isinstance(r, np.ndarray)

    def test_accepts_ndarray(self):
        r = rank_alternatives(np.array([3.0, 1.0, 2.0]))
        assert isinstance(r, np.ndarray)

    def test_accepts_tuple(self):
        r = rank_alternatives((3, 1, 2))
        assert isinstance(r, np.ndarray)

    def test_best_gets_rank_one_descending(self):
        scores = [10, 50, 30, 20]
        r = rank_alternatives(scores, descending=True)
        best_idx = np.argmax(scores)
        assert r[best_idx] == 1.0

    def test_best_gets_rank_one_ascending(self):
        scores = [10, 50, 30, 20]
        r = rank_alternatives(scores, descending=False)
        best_idx = np.argmin(scores)
        assert r[best_idx] == 1.0

    def test_all_ranks_present_no_ties(self):
        scores = [5, 3, 8, 1, 9]
        r = rank_alternatives(scores)
        assert sorted(r.tolist()) == [1.0, 2.0, 3.0, 4.0, 5.0]


# Tie-breaking: 'average' 
class TestAverageMethod:

    def test_two_way_tie(self):
        scores = [0.5, 0.5, 0.3]
        r = rank_alternatives(scores, method='average')
        # both 0.5s tie for rank 1-2 -> average 1.5, 0.3 gets rank 3
        assert r.tolist() == [1.5, 1.5, 3.0]

    def test_three_way_tie(self):
        scores = [0.5, 0.5, 0.5]
        r = rank_alternatives(scores, method='average')
        # all tie for ranks 1,2,3 -> average = 2.0
        assert np.allclose(r, [2.0, 2.0, 2.0])

    def test_four_way_tie_with_outlier(self):
        scores = [1, 1, 1, 1, 5]
        r = rank_alternatives(scores, method='average')
        # 5 is best -> rank 1; four 1's tie for ranks 2,3,4,5 -> avg 3.5
        assert r[4] == 1.0
        assert np.allclose(r[:4], 3.5)

    def test_partial_ties(self):
        scores = [10, 10, 5, 5, 5]
        r = rank_alternatives(scores, method='average')
        # two 10's tie for rank 1,2 -> avg 1.5
        # three 5's tie for rank 3,4,5 -> avg 4.0
        assert np.allclose(r[:2], 1.5)
        assert np.allclose(r[2:], 4.0)

    def test_average_matches_legacy_two_way(self):
        """Legacy rank() and new rank_alternatives() must agree for <=2-way ties."""
        scores = np.array([0.5, 0.5, 0.3])
        old = legacy_rank(scores)
        new = rank_alternatives(scores, method='average')
        assert np.allclose(old, new)

    def test_average_fixes_legacy_bug_for_3way_tie(self):
        """
        The legacy implementation had a formula bug for tie groups > 2.
        For three tied values at ranks 1,2,3 the correct average is 2.0.
        """
        scores = np.array([0.5, 0.5, 0.5])
        new = rank_alternatives(scores, method='average')
        assert np.allclose(new, [2.0, 2.0, 2.0])


#  Tie-breaking: 'min' / 'max' / 'dense' / 'ordinal'
class TestOtherTieMethods:

    def test_min_method(self):
        scores = [10, 10, 8]
        r = rank_alternatives(scores, method='min')
        # competition ranking: "1 1 3"
        assert r.tolist() == [1, 1, 3]

    def test_max_method(self):
        scores = [10, 10, 8]
        r = rank_alternatives(scores, method='max')
        # "2 2 3"
        assert r.tolist() == [2, 2, 3]

    def test_dense_method(self):
        scores = [10, 10, 8, 6, 6]
        r = rank_alternatives(scores, method='dense')
        # distinct values: 10 (rank1), 8 (rank2), 6 (rank3) -> "1 1 2 3 3"
        assert r.tolist() == [1, 1, 2, 3, 3]

    def test_ordinal_method_breaks_all_ties(self):
        scores = [5, 5, 5]
        r = rank_alternatives(scores, method='ordinal')
        # every alternative gets a unique rank
        assert sorted(r.tolist()) == [1, 2, 3]
        assert len(set(r.tolist())) == 3

    def test_min_dtype_is_int(self):
        r = rank_alternatives([1, 2, 3], method='min')
        assert np.issubdtype(r.dtype, np.integer)

    def test_average_dtype_is_float(self):
        r = rank_alternatives([1, 2, 3], method='average')
        assert np.issubdtype(r.dtype, np.floating)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match='Unknown tie-breaking method'):
            rank_alternatives([1, 2, 3], method='bogus')


# Validation / error handling 
class TestValidation:

    def test_empty_raises(self):
        with pytest.raises(ValueError, match='at least one'):
            rank_alternatives([])

    def test_2d_array_raises(self):
        with pytest.raises(ValueError, match='1-D'):
            rank_alternatives(np.ones((3, 2)))

    def test_nan_raises(self):
        with pytest.raises(ValueError, match='NaN'):
            rank_alternatives([1.0, np.nan, 3.0])

    def test_inf_raises(self):
        with pytest.raises(ValueError, match='Inf'):
            rank_alternatives([1.0, np.inf, 3.0])

    def test_non_numeric_raises_typeerror(self):
        with pytest.raises(TypeError):
            rank_alternatives(['a', 'b', 'c'])

    def test_legacy_rank_raises_valueerror_on_bad_input(self):
        """helpers.rank() must still raise ValueError (not crash silently) on bad input."""
        with pytest.raises(ValueError):
            legacy_rank([1.0, np.nan, 3.0])

    def test_legacy_rank_raises_on_empty(self):
        with pytest.raises(ValueError):
            legacy_rank([])


# Import paths / re-exports
class TestImportPaths:
    def test_import_from_methods_utils_star(self):
        from pyfdm.methods.utils import rank_alternatives as ra
        assert callable(ra)

    def test_import_from_methods_top_level(self):
        from pyfdm.methods import rank_alternatives as ra
        assert callable(ra)

    def test_legacy_helpers_rank_still_works(self):
        r = legacy_rank(np.array([1.0, 2.0, 3.0]))
        assert isinstance(r, np.ndarray)


# Integration with BaseFuzzyMethod.rank()
class TestBaseFuzzyMethodIntegration:

    @pytest.fixture
    def matrix(self):
        return np.array([
            [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
            [(2, 3, 4), (5, 6, 7), (1, 2, 3)],
            [(3, 4, 5), (2, 3, 4), (4, 5, 6)],
        ], dtype=float)

    @pytest.fixture
    def weights(self):
        return np.array([0.4, 0.35, 0.25])

    @pytest.fixture
    def types(self):
        return np.array([1, -1, 1])

    def test_base_rank_matches_standalone_rank(self, matrix, weights, types):
        from pyfdm.methods import fTOPSIS
        m = fTOPSIS()
        prefs = m(matrix, weights, types)
        method_rank = m.rank()
        standalone_rank = rank_alternatives(prefs, descending=True, method='average')
        assert np.allclose(method_rank, standalone_rank)

    def test_base_rank_respects_descending_flag(self, matrix, weights, types):
        from pyfdm.methods import fPIV   # _descending = False (lower is better)
        m = fPIV()
        prefs = m(matrix, weights, types)
        method_rank = m.rank()
        standalone_rank = rank_alternatives(prefs, descending=False, method='average')
        assert np.allclose(method_rank, standalone_rank)

    def test_rank_alternatives_can_post_process_any_external_scores(self):
        """
        Core use case: rank a vector of scores that did NOT come from
        running a pyfdm MCDA method (e.g. expert scoring, external model).
        """
        external_scores = np.array([72.5, 88.0, 88.0, 60.1, 95.3])
        r = rank_alternatives(external_scores, descending=True)
        assert r[4] == 1.0          # 95.3 is the best
        assert r[3] == 5.0          # 60.1 is the worst
        assert r[1] == r[2] == 2.5  # the two 88.0 ties average to 2.5


# Larger / randomised sanity checks 
class TestSanity:

    def test_rank_is_permutation_for_no_ties(self):
        rng = np.random.default_rng(42)
        scores = rng.permutation(20).astype(float)
        r = rank_alternatives(scores, method='average')
        assert sorted(r.tolist()) == [float(i) for i in range(1, 21)]

    def test_descending_and_ascending_are_consistent(self):
        scores = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        r_desc = rank_alternatives(scores, descending=True)
        r_asc = rank_alternatives(scores, descending=False)
        # rank under descending + rank under ascending should sum to n+1 for each (no ties case)
        n = len(scores)
        assert np.allclose(r_desc + r_asc, n + 1)

    def test_large_input_performance(self):
        """Should run efficiently even for larger n (no O(n^2) blowup)."""
        rng = np.random.default_rng(0)
        scores = rng.random(2000)
        r = rank_alternatives(scores)
        assert r.shape == (2000,)
        assert r.min() == 1.0
        assert r.max() == 2000.0
