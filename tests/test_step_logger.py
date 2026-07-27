# Copyright (c) 2026 Jakub Więckowski

import pytest
import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyfdm.methods import *
from pyfdm.step_logger import StepLogger


# Fixtures
@pytest.fixture
def matrix():
    return np.array([
        [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
        [(2, 3, 4), (5, 6, 7), (1, 2, 3)],
        [(3, 4, 5), (2, 3, 4), (4, 5, 6)],
    ], dtype=float)

@pytest.fixture
def weights():
    return np.array([0.4, 0.35, 0.25])

@pytest.fixture
def types():
    return np.array([1, -1, 1])

@pytest.fixture
def alt_names():
    return ['Alpha', 'Beta', 'Gamma']

@pytest.fixture
def crit_names():
    return ['Cost', 'Quality', 'Speed']


# StepLogger construction 

class TestStepLoggerConstruction:

    def test_single_format_string(self):
        lg = StepLogger(output='console')
        assert 'console' in lg.output

    def test_multiple_formats_list(self):
        lg = StepLogger(output=['console', 'json'])
        assert 'console' in lg.output
        assert 'json' in lg.output

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match='Invalid output format'):
            StepLogger(output='pdf')

    def test_invalid_format_in_list_raises(self):
        with pytest.raises(ValueError, match='Invalid output format'):
            StepLogger(output=['console', 'xml'])

    def test_all_valid_formats_accepted(self):
        lg = StepLogger(output=['console', 'json', 'csv', 'excel'])
        assert lg.output == {'console', 'json', 'csv', 'excel'}

    def test_default_path_generated(self):
        lg = StepLogger(output='json')
        assert 'pyfdm_steps_' in str(lg.path)

    def test_custom_path_accepted(self):
        lg = StepLogger(output='json', path='/tmp/my_results')
        assert str(lg.path) == '/tmp/my_results'

    def test_custom_labels(self, alt_names, crit_names):
        lg = StepLogger(output='console', alternative_names=alt_names, criterion_names=crit_names)
        assert lg.alt_names == alt_names
        assert lg.crit_names == crit_names


# StepLogger.log() and internals 

class TestStepLoggerLog:

    def test_log_records_step(self):
        lg = StepLogger(output='console')
        lg.log('Test step', np.array([1.0, 2.0, 3.0]), 'A test description')
        assert len(lg.steps) == 1
        assert lg.steps[0]['name'] == 'Test step'

    def test_log_multiple_steps_ordered(self):
        lg = StepLogger(output='console')
        for i in range(5):
            lg.log(f'Step {i}', np.ones(3) * i)
        assert len(lg.steps) == 5
        assert lg.steps[2]['name'] == 'Step 2'

    def test_clear_removes_all_steps(self):
        lg = StepLogger(output='console')
        lg.log('A', np.ones(3))
        lg.log('B', np.ones(3))
        lg.clear()
        assert len(lg.steps) == 0

    def test_log_scalar_value(self):
        lg = StepLogger(output='console')
        lg.log('Lambda', 0.5)
        assert lg.steps[0]['data'] == 0.5

    def test_log_dict_value(self):
        lg = StepLogger(output='console')
        lg.log('Multi', {'S': np.ones(3), 'R': np.zeros(3)})
        assert isinstance(lg.steps[0]['data'], dict)


# verbose parameter on methods

class TestVerboseParameter:

    def test_verbose_none_no_steps(self, matrix, weights, types):
        method = fTOPSIS()
        method(matrix, weights, types, verbose=None)
        assert not hasattr(method, '_logger')   # no logger attached

    def test_verbose_step_logger_populates_steps(self, matrix, weights, types):
        lg = StepLogger(output='console')
        fTOPSIS()(matrix, weights, types, verbose=lg)
        assert len(lg.steps) > 0

    def test_verbose_wrong_type_raises(self, matrix, weights, types):
        with pytest.raises(TypeError, match='StepLogger'):
            fTOPSIS()(matrix, weights, types, verbose='console')

    def test_verbose_wrong_type_bool_raises(self, matrix, weights, types):
        with pytest.raises(TypeError, match='StepLogger'):
            fTOPSIS()(matrix, weights, types, verbose=True)

    def test_verbose_method_name_set(self, matrix, weights, types):
        lg = StepLogger(output='console')
        fTOPSIS()(matrix, weights, types, verbose=lg)
        assert lg._method_name == 'fTOPSIS'

    def test_result_unchanged_with_verbose(self, matrix, weights, types):
        """verbose must not alter the returned preferences."""
        lg = StepLogger(output='console')
        prefs_verbose = fTOPSIS()(matrix, weights, types, verbose=lg)
        prefs_plain   = fTOPSIS()(matrix, weights, types)
        assert np.allclose(prefs_verbose, prefs_plain)

    def test_ranking_unchanged_with_verbose(self, matrix, weights, types):
        lg = StepLogger(output='console')
        m = fTOPSIS()
        m(matrix, weights, types, verbose=lg)
        rank_v = m.rank()
        m2 = fTOPSIS()
        m2(matrix, weights, types)
        rank_p = m2.rank()
        assert np.array_equal(rank_v, rank_p)


# Step content per method 

class TestStepContent:
    """Each method must log a minimum set of named steps."""

    def _get_steps(self, method_cls, matrix, weights, types, **kwargs):
        lg = StepLogger(output='console')
        method_cls()( matrix, weights, types, verbose=lg, **kwargs)
        return {s['name'] for s in lg.steps}

    def test_topsis_has_normalised_matrix(self, matrix, weights, types):
        steps = self._get_steps(fTOPSIS, matrix, weights, types)
        assert 'Normalized matrix' in steps

    def test_topsis_has_weighted_matrix(self, matrix, weights, types):
        steps = self._get_steps(fTOPSIS, matrix, weights, types)
        assert 'Weighted normalized matrix' in steps

    def test_topsis_has_fpis_fnis(self, matrix, weights, types):
        steps = self._get_steps(fTOPSIS, matrix, weights, types)
        assert 'Fuzzy Positive Ideal Solution (FPIS)' in steps
        assert 'Fuzzy Negative Ideal Solution (FNIS)' in steps

    def test_topsis_has_distances(self, matrix, weights, types):
        steps = self._get_steps(fTOPSIS, matrix, weights, types)
        assert 'Distance to FPIS' in steps
        assert 'Distance to FNIS' in steps

    def test_topsis_has_preferences(self, matrix, weights, types):
        steps = self._get_steps(fTOPSIS, matrix, weights, types)
        assert 'Preference scores (CC)' in steps

    def test_vikor_has_ideal_nadir(self, matrix, weights, types):
        steps = self._get_steps(fVIKOR, matrix, weights, types)
        assert 'Ideal solution (f*)' in steps
        assert 'Nadir solution (f-)' in steps

    def test_vikor_has_S_R_Q(self, matrix, weights, types):
        steps = self._get_steps(fVIKOR, matrix, weights, types)
        assert 'S vector (utility measure)' in steps
        assert 'R vector (regret measure)' in steps

    def test_moora_has_profit_cost_sums(self, matrix, weights, types):
        steps = self._get_steps(fMOORA, matrix, weights, types)
        assert 'Profit sum (Sp)' in steps

    def test_edas_has_pda_nda(self, matrix, weights, types):
        steps = self._get_steps(fEDAS, matrix, weights, types)
        assert 'Positive Distance from Average (PDA)' in steps
        assert 'Negative Distance from Average (NDA)' in steps

    def test_mabac_has_border_area(self, matrix, weights, types):
        steps = self._get_steps(fMABAC, matrix, weights, types)
        assert 'Border Approximation Area (G)' in steps

    def test_waspas_has_wsm_wpm(self, matrix, weights, types):
        steps = self._get_steps(fWASPAS, matrix, weights, types)
        assert 'WSM weighted matrix' in steps
        assert 'WPM weighted matrix' in steps

    def test_marcos_has_utility_functions(self, matrix, weights, types):
        lg = StepLogger(output='console')
        fMARCOS()(matrix, weights, types, verbose=lg)
        names = {s['name'] for s in lg.steps}
        assert any('AAI' in n or 'AI' in n or 'Utility' in n or 'Ideal' in n or 'Normalised' in n for n in names)


# Console output 

class TestConsoleOutput:

    def test_console_output_printed(self, matrix, weights, types, capsys):
        lg = StepLogger(output='console')
        fTOPSIS()(matrix, weights, types, verbose=lg)
        captured = capsys.readouterr()
        assert 'fTOPSIS' in captured.out

    def test_console_shows_step_names(self, matrix, weights, types, capsys):
        lg = StepLogger(output='console')
        fTOPSIS()(matrix, weights, types, verbose=lg)
        captured = capsys.readouterr()
        assert 'Normalized matrix' in captured.out
        assert 'Preference scores' in captured.out

    def test_console_shows_custom_alt_names(self, matrix, weights, types, capsys, alt_names):
        lg = StepLogger(output='console', alternative_names=alt_names)
        fTOPSIS()(matrix, weights, types, verbose=lg)
        captured = capsys.readouterr()
        assert 'Alpha' in captured.out

    def test_manual_print(self, matrix, weights, types, capsys):
        """logger.print() should re-print even without console in output."""
        lg = StepLogger(output='json', path='/tmp/test_manual')
        fTOPSIS()(matrix, weights, types, verbose=lg)
        lg.print()
        captured = capsys.readouterr()
        assert 'fTOPSIS' in captured.out


# JSON output 

class TestJSONOutput:

    def test_json_file_created(self, tmp_path, matrix, weights, types):
        path = tmp_path / 'steps'
        lg = StepLogger(output='json', path=path)
        fTOPSIS()(matrix, weights, types, verbose=lg)
        assert (tmp_path / 'steps.json').exists()

    def test_json_has_method_name(self, tmp_path, matrix, weights, types):
        path = tmp_path / 'steps'
        lg = StepLogger(output='json', path=path)
        fTOPSIS()(matrix, weights, types, verbose=lg)
        data = json.loads((tmp_path / 'steps.json').read_text())
        assert data['method'] == 'fTOPSIS'

    def test_json_has_all_steps(self, tmp_path, matrix, weights, types):
        path = tmp_path / 'steps'
        lg = StepLogger(output='json', path=path)
        fTOPSIS()(matrix, weights, types, verbose=lg)
        data = json.loads((tmp_path / 'steps.json').read_text())
        assert len(data['steps']) >= 5

    def test_json_step_has_name_description_data(self, tmp_path, matrix, weights, types):
        path = tmp_path / 'steps'
        lg = StepLogger(output='json', path=path)
        fTOPSIS()(matrix, weights, types, verbose=lg)
        data = json.loads((tmp_path / 'steps.json').read_text())
        step = data['steps'][0]
        assert 'name' in step
        assert 'description' in step
        assert 'data' in step

    def test_json_3d_matrix_stored_as_tfn(self, tmp_path, matrix, weights, types):
        path = tmp_path / 'steps'
        lg = StepLogger(output='json', path=path)
        fTOPSIS()(matrix, weights, types, verbose=lg)
        data = json.loads((tmp_path / 'steps.json').read_text())
        # find a step with ndim=3
        tfn_steps = [s for s in data['steps'] if isinstance(s['data'], dict) and 'tfn' in s['data']]
        assert len(tfn_steps) > 0

    def test_json_1d_array_has_values(self, tmp_path, matrix, weights, types):
        path = tmp_path / 'steps'
        lg = StepLogger(output='json', path=path)
        fTOPSIS()(matrix, weights, types, verbose=lg)
        data = json.loads((tmp_path / 'steps.json').read_text())
        # Preference scores are 1D
        pref_step = next(s for s in data['steps'] if 'Preference' in s['name'])
        assert 'values' in pref_step['data']
        assert len(pref_step['data']['values']) == 3


# CSV output 

class TestCSVOutput:

    def test_csv_file_created(self, tmp_path, matrix, weights, types):
        path = tmp_path / 'steps'
        lg = StepLogger(output='csv', path=path)
        fTOPSIS()(matrix, weights, types, verbose=lg)
        assert (tmp_path / 'steps.csv').exists()

    def test_csv_contains_method_name(self, tmp_path, matrix, weights, types):
        path = tmp_path / 'steps'
        lg = StepLogger(output='csv', path=path)
        fTOPSIS()(matrix, weights, types, verbose=lg)
        content = (tmp_path / 'steps.csv').read_text()
        assert 'fTOPSIS' in content

    def test_csv_contains_step_names(self, tmp_path, matrix, weights, types):
        path = tmp_path / 'steps'
        lg = StepLogger(output='csv', path=path)
        fTOPSIS()(matrix, weights, types, verbose=lg)
        content = (tmp_path / 'steps.csv').read_text()
        assert 'Normalized matrix' in content

    def test_csv_has_alternative_labels(self, tmp_path, matrix, weights, types, alt_names):
        path = tmp_path / 'steps'
        lg = StepLogger(output='csv', path=path, alternative_names=alt_names)
        fTOPSIS()(matrix, weights, types, verbose=lg)
        content = (tmp_path / 'steps.csv').read_text()
        assert 'Alpha' in content

    def test_csv_3d_tfn_expanded_to_lmu_columns(self, tmp_path, matrix, weights, types):
        path = tmp_path / 'steps'
        lg = StepLogger(output='csv', path=path)
        fTOPSIS()(matrix, weights, types, verbose=lg)
        content = (tmp_path / 'steps.csv').read_text()
        assert '_l' in content and '_m' in content and '_u' in content

    def test_csv_multiple_steps_separated_by_blank(self, tmp_path, matrix, weights, types):
        path = tmp_path / 'steps'
        lg = StepLogger(output='csv', path=path)
        fTOPSIS()(matrix, weights, types, verbose=lg)
        content = (tmp_path / 'steps.csv').read_text()
        # blank rows between steps
        assert '\n\n' in content or ',\n,' in content


# Excel output 

class TestExcelOutput:

    def test_excel_file_created(self, tmp_path, matrix, weights, types):
        pytest.importorskip('openpyxl')
        path = tmp_path / 'steps'
        lg = StepLogger(output='excel', path=path)
        fTOPSIS()(matrix, weights, types, verbose=lg)
        assert (tmp_path / 'steps.xlsx').exists()

    def test_excel_has_overview_sheet(self, tmp_path, matrix, weights, types):
        openpyxl = pytest.importorskip('openpyxl')
        path = tmp_path / 'steps'
        lg = StepLogger(output='excel', path=path)
        fTOPSIS()(matrix, weights, types, verbose=lg)
        wb = openpyxl.load_workbook(tmp_path / 'steps.xlsx')
        assert 'Overview' in wb.sheetnames

    def test_excel_has_sheet_per_step(self, tmp_path, matrix, weights, types):
        openpyxl = pytest.importorskip('openpyxl')
        path = tmp_path / 'steps'
        lg = StepLogger(output='excel', path=path)
        fTOPSIS()(matrix, weights, types, verbose=lg)
        wb = openpyxl.load_workbook(tmp_path / 'steps.xlsx')
        # overview + n step sheets
        assert len(wb.sheetnames) > 1

    def test_excel_overview_contains_method_name(self, tmp_path, matrix, weights, types):
        openpyxl = pytest.importorskip('openpyxl')
        path = tmp_path / 'steps'
        lg = StepLogger(output='excel', path=path)
        fTOPSIS()(matrix, weights, types, verbose=lg)
        wb = openpyxl.load_workbook(tmp_path / 'steps.xlsx')
        ws = wb['Overview']
        cell_values = [ws.cell(row=r, column=c).value
                       for r in range(1, 5) for c in range(1, 4)]
        assert any('fTOPSIS' in str(v) for v in cell_values if v)

    def test_excel_missing_openpyxl_raises(self, tmp_path, matrix, weights, types, monkeypatch):
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == 'openpyxl':
                raise ImportError('no openpyxl')
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, '__import__', mock_import)

        path = tmp_path / 'steps'
        lg = StepLogger(output='excel', path=path)
        lg._set_method('fTOPSIS')
        lg.log('test', np.ones(3))
        with pytest.raises(ImportError, match='openpyxl'):
            lg._write_excel()


# Reuse logger across multiple runs 

class TestLoggerReuse:

    def test_clear_and_rerun(self, matrix, weights, types):
        lg = StepLogger(output='console')
        fTOPSIS()(matrix, weights, types, verbose=lg)
        n_steps_first = len(lg.steps)
        lg.clear()
        fTOPSIS()(matrix, weights, types, verbose=lg)
        assert len(lg.steps) == n_steps_first

    def test_different_methods_accumulate_steps_after_clear(self, matrix, weights, types):
        lg = StepLogger(output='console')
        fTOPSIS()(matrix, weights, types, verbose=lg)
        lg.clear()
        fMOORA()(matrix, weights, types, verbose=lg)
        assert lg._method_name == 'fMOORA'
        assert len(lg.steps) > 0


# Integration 

class TestVerboseIntegration:

    def test_full_workflow_with_all_outputs(self, tmp_path, matrix, weights, types):
        pytest.importorskip('openpyxl')
        path = tmp_path / 'full'
        lg = StepLogger(
            output=['console', 'json', 'csv', 'excel'],
            path=path,
            alternative_names=['Car A', 'Car B', 'Car C'],
            criterion_names=['Price', 'Quality', 'Speed'],
        )
        prefs = fTOPSIS()(matrix, weights, types, verbose=lg)
        assert prefs.shape == (3,)
        assert (tmp_path / 'full.json').exists()
        assert (tmp_path / 'full.csv').exists()
        assert (tmp_path / 'full.xlsx').exists()

    def test_cocoas_vikor_special_steps(self, matrix, weights, types):
        """Methods with more than basic steps log all of them."""
        lg = StepLogger(output='console')
        fVIKOR()(matrix, weights, types, verbose=lg)
        names = {s['name'] for s in lg.steps}
        assert 'Crisp Q (compromise score)' in names

    def test_edas_appraisal_score_present(self, matrix, weights, types):
        lg = StepLogger(output='console')
        fEDAS()(matrix, weights, types, verbose=lg)
        names = {s['name'] for s in lg.steps}
        assert 'Appraisal Score (AS)' in names
