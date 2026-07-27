# Copyright (c) 2026 Jakub Więckowski

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyfdm.expert.scales import SCALE_1_5, SCALE_1_7, SCALE_1_9, get_tfn

class TestScales:

    def test_get_tfn_scale_1_9(self):
        tfn = get_tfn(5, SCALE_1_9)
        assert tfn.shape == (3,)
        assert tfn[0] <= tfn[1] <= tfn[2]

    def test_get_tfn_scale_1_5(self):
        for rating in SCALE_1_5:
            tfn = get_tfn(rating, SCALE_1_5)
            assert tfn[0] <= tfn[1] <= tfn[2]

    def test_get_tfn_scale_1_7(self):
        for rating in SCALE_1_7:
            tfn = get_tfn(rating, SCALE_1_7)
            assert tfn[0] <= tfn[1] <= tfn[2]

    def test_invalid_rating_raises(self):
        with pytest.raises(ValueError, match='not in scale'):
            get_tfn(10, SCALE_1_9)
