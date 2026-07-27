# Copyright (c) 2026 Jakub Więckowski

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyfdm.expert import FuzzyExporter, ExpertCollector
from pyfdm.expert.scales import SCALE_1_9

class TestFuzzyExporter:

    @pytest.fixture
    def populated_collector(self):
        c = ExpertCollector(n_alternatives=3, n_criteria=3, scale=SCALE_1_9)
        c.add_expert_matrix(0, [[9, 3, 7], [5, 9, 3], [3, 5, 9]])
        c.add_expert_matrix(1, [[7, 5, 9], [9, 3, 5], [5, 7, 3]])
        return c

    def test_to_csv(self, tmp_path, populated_collector):
        path = tmp_path / 'test.csv'
        FuzzyExporter(populated_collector).to_csv(path)
        assert path.exists()
        content = path.read_text()
        assert 'Expert' in content
        assert 'A1' in content

    def test_to_json(self, tmp_path, populated_collector):
        import json
        path = tmp_path / 'test.json'
        FuzzyExporter(populated_collector).to_json(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert 'experts' in data
        assert data['metadata']['n_experts'] == 2

    def test_to_json_structure(self, tmp_path, populated_collector):
        import json
        path = tmp_path / 'test.json'
        FuzzyExporter(populated_collector).to_json(path)
        data = json.loads(path.read_text())
        assert '0' in data['experts']
        assert 'A1' in data['experts']['0']

    def test_to_csv_has_correct_columns(self, tmp_path, populated_collector):
        path = tmp_path / 'test.csv'
        FuzzyExporter(populated_collector).to_csv(path)
        header = path.read_text().split('\n')[0]
        assert 'C1_l' in header and 'C1_m' in header and 'C1_u' in header

    def test_to_excel_no_openpyxl(self, tmp_path, populated_collector, monkeypatch):
        """Should raise ImportError when openpyxl missing."""
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == 'openpyxl':
                raise ImportError('openpyxl not installed')
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, '__import__', mock_import)
        with pytest.raises(ImportError, match='openpyxl'):
            FuzzyExporter(populated_collector).to_excel(tmp_path / 'test.xlsx')
