# Changelog

All notable changes to pyfdm are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.2.0] — 2026

### Added

**Core architecture**

- `BaseFuzzyMethod` abstract base class (`pyfdm.methods.BaseFuzzyMethod`): all 21 MCDA method classes now inherit from it, providing a shared `rank()` implementation, consistent `__repr__`, and a `method_name` property.
- `rank_alternatives()` standalone ranking utility (`pyfdm.utils.rank_alternatives`): computes a ranking from any 1-D score vector with five tie-breaking strategies — `'average'` (default, identical to old `helpers.rank()` behaviour), `'min'`, `'max'`, `'dense'`, `'ordinal'`. Full input validation: raises on NaN, Inf, empty arrays, non-1-D input, non-numeric data. Runs in O(n log n).
- `StepLogger` class (`pyfdm.step_logger.StepLogger`): captures every intermediate matrix and vector computed by an MCDA method or subjective weighting method. Activated by passing a `StepLogger` instance as `logger=` in any method's `__init__`. Supports console tables, JSON, CSV, and Excel output, individually or combined.
- `logger=` keyword argument added to `__init__` on all 21 MCDA method classes. Accepts a `StepLogger` or `None`; raises `TypeError` on any other value.
- `intermediate_results` now are included in both fuzzy MCDA methods and fuzzy subjective weighting methods. Can be used to verify and check partial results from the calculation process.

**Validator improvements**

- `validate_matrix_shape`: now explicitly checks `m >= 2` alternatives and `n >= 1` criteria.
- `validate_tfn`: new check — every TFN `(l, m, u)` must satisfy `l <= m <= u`; NaN and Inf are rejected with descriptive messages.
- `validate_weights`: new check — 1-D weights must be non-negative; 2-D TFN weights must each satisfy `l <= m <= u`.
- `validate_types`: new check — values must be in `{1, −1}`.
- `validate_input`: explicit dimension-mismatch check between matrix, weights, and types.
- `validate_comparison_matrix`: new method for subjective weighting inputs (square `(n, n, 3)` TFN matrices).
- `validate_param_range`: new method to check if given parameter is in the required range.
- `validate_vectors`: new method to check rankings given as parameters for correlation and similarity coefficients.
- several functions for method-specific input data validation: `validate_ervd_input`, `validate_rim_input`, `validate_rafsi_input`, `validate_spotis_input`, `validate_lmaw_input`, `validate_bwm_input`

**New MCDA methods (6)**

- `fAROMAN` — fuzzy Alternative Ranking Order Method Accounting for two-step Normalization.
- `fERVD` — fuzzy Election based on Relative Value Distances. Supports custom reference points.
- `fMARCOS` — fuzzy Measurement of Alternatives and Ranking according to Compromise Solution.
- `fPIV` — fuzzy Proximity Indexed Value.
- `fRAFSI` — Fuzzy Ranking of Alternatives through Functional mapping of criterion Sub-Intervals into a Single Interval.
- `fRIM` — Fuzzy Reference Ideal Method.

**Subjective weighting methods (6)** — `pyfdm.weights.subjective`

- `fAHP` — Fuzzy Analytic Hierarchy Process with optional Consistency Ratio check.
- `fBWM` — Fuzzy Best-Worst Method with iterative gradient-descent solver.
- `fFUCOM` — Fuzzy Full Consistency Method.
- `fLMAW` — Fuzzy Logarithm Methodology of Additive Weights.
- `fRANCOM` — Fuzzy RANking COMparison; accepts tied rankings.
- `fSWARA` — Fuzzy Step-wise Weight Assessment Ratio Analysis.

**Objective weights reorganization** — `pyfdm.weights.objective`

- Existing functions (`equal_weights`, `shannon_entropy_weights`, `standard_deviation_weights`, `variance_weights`) are now also available under `pyfdm.weights.objective`. Direct imports from `pyfdm.weights` continue to work.

**Expert data collection and export** — `pyfdm.expert`

- `ExpertCollector`: collects per-expert integer ratings, validates them against a linguistic scale, and converts to TFN decision matrices.
- `FuzzyExporter`: exports collected data to CSV, JSON, or Excel (requires `openpyxl`).
- Built-in linguistic scales: `SCALE_1_5`, `SCALE_1_7`, `SCALE_1_9`.
- `get_tfn(rating, scale)` helper for single-value conversion.

**Group aggregation** — `pyfdm.group`

- `aggregate(matrices, method=...)`: geometric mean (recommended for fAHP), arithmetic mean, weighted average, OWA.
- Individual functions exported: `geometric_mean`, `arithmetic_mean`, `weighted_average`, `owa`.

**TFN utilities** — `pyfdm.utils`

- `normalize_weights(weights)`: moved here from `pyfdm.helpers`; old location emits `DeprecationWarning`.
- `crisp_to_tfn_weights(weights)`: expands crisp weights to degenerate TFNs `(w, w, w)`.
- `tfn_to_crisp(weights, method)`: defuzzifies TFN weights; supports `'graded_mean'`, `'mean'`, `'mean_area'`, `'centroid'`.
- `defuzzify_matrix(matrix, method)`: defuzzifies a full `(m, n, 3)` TFN matrix to crisp `(m, n)`.

**Packaging**

- `setup.py`: bumped to version `1.2.0`; added `extras_require = {'excel': ['openpyxl'], 'dev': ['pytest', 'openpyxl']}`.
- `docs/conf.py`: version updated to `1.2.0`, copyright year updated.
- `CHANGELOG.md`: this file added.

### Changed

- `helpers.rank()` now delegates to `rank_alternatives(method='average')`. Behaviour is identical for the default `descending=True` case; a three-way tie now correctly returns `2.0` (average of ranks 1, 2, 3) rather than the previously incorrect value from a broken formula.
- `BaseFuzzyMethod.rank()` delegates to `rank_alternatives` directly instead of re-importing from `helpers`, removing a circular-import risk.
- Sphinx `docs/`: all new packages, submodules, and methods have corresponding `.rst` files; `pyfdm.rst` and `pyfdm.methods.rst` updated to include them in the toctree.

### Deprecated

- `pyfdm.helpers.normalize_weights`: use `from pyfdm.utils import normalize_weights` instead. The old import path raises `DeprecationWarning` and will be removed in v1.3.0.

## [1.1.13] — 2024

Last stable release before the v1.2.0 overhaul. Contains 15 MCDA methods (ARAS, COCOSO, CODAS, COPRAS, EDAS, MABAC, MAIRCA, MOORA, OCRA, SPOTIS, TOPSIS, VIKOR, WASPAS, WPM, WSM), 4 objective weighting functions, 4 correlation coefficients, TFN arithmetic, and Sphinx documentation.
