# pyfdm

Python 3 package with Fuzzy Decision Making (PyFDM) methods based on Triangular Fuzzy Numbers (TFN).

**Version 1.2.1** adds 6 new MCDA methods, 6 subjective weighting methods, an expert-data collection and export module, group aggregation, step-by-step intermediate result inspection, and a standalone ranking utility — on top of a refactored, more robust core architecture. See [What's new in 1.2.1](#whats-new-in-120) below.

---

# Table of contents

- [Installation](#installation)
- [Local development setup](#local-development-setup)
- [Testing](#testing)
- [What's new in 1.2.1](#whats-new-in-120)
- [Modules and functionalities](#modules-and-functionalities)
- [Usage examples](#usage-examples)
- [Citations](#citations)
- [References](#references)

---

# Installation

The package can be installed from PyPI with pip:

```bash
pip install pyfdm
```

Excel-based features (`FuzzyExporter.to_excel`, `StepLogger(output='excel')`) require the optional `openpyxl` dependency. Install it together with the package using the `excel` extra:

```bash
pip install pyfdm[excel]
```

or separately:

```bash
pip install openpyxl
```

---

# Local development setup

To work on the library itself (contribute new methods, fix bugs, run the test suite against your changes), clone the repository and install it in editable mode.

### 1. Clone the repository

```bash
git clone https://github.com/jwieckowski/pyfdm.git
cd pyfdm
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install in editable mode with development dependencies

```bash
pip install -e ".[dev]"
```

This installs the package itself in editable mode (so source edits take effect immediately without reinstalling), plus `pytest` and `openpyxl` for running the full test suite, including Excel-export tests.

If you only need the core runtime dependencies without the dev extras:

```bash
pip install -e .
```

### 4. Verify the installation

```bash
python -c "import pyfdm; print(pyfdm.__file__)"
```

This should print a path inside your local repository clone, confirming the editable install is active.

---

# Testing

The test suite uses `pytest` and covers the core MCDA methods, validators, weighting methods, the expert/export module, group aggregation, sensitivity analysis, the step-by-step logger, and the standalone ranking utility.

Install test dependencies (if not already installed via `pip install -e ".[dev]"`):

```bash
pip install pytest openpyxl
```

Run the full suite from the repository root:

```bash
pytest tests
```

Run a specific test file:

```bash
pytest tests/test_methods.py -v
pytest tests/test_step_logger.py -v
pytest tests/test_ranking.py -v
```

Run with a short traceback (useful while developing):

```bash
pytest tests -v --tb=short
```

All tests are expected to pass with zero `DeprecationWarning`s on a clean checkout.

---

# What's new in 1.2.1

### Architecture

- **`BaseFuzzyMethod`** — a shared abstract base class for all MCDA methods (`pyfdm.methods.BaseFuzzyMethod`). Eliminates duplicated `rank()` logic across 21 method classes and gives every method a consistent `__repr__`.
- **`StepLogger`** class (`pyfdm.step_logger.StepLogger`): captures every intermediate matrix and vector computed by an MCDA method or subjective weighting method. Activated by passing a `StepLogger` instance as `logger=` in any method's `__init__`. Supports console tables, JSON, CSV, and Excel output, individually or combined.
- **`intermediate_results`** now are included in both fuzzy MCDA methods and fuzzy subjective weighting methods. Can be used to verify and check partial results from the calculation process.
- **Extended `Validator`** — decision matrices, weights, and criteria types are now validated for shape, non-negativity, `l <= m <= u` TFN consistency, and absence of `NaN`/`Inf`, with descriptive error messages.
- **Fixed `validate_types` bug** — matrices where all criteria share the same type (all-profit or all-cost) are no longer incorrectly rejected.

### New MCDA methods (6)

`fAROMAN`, `fERVD`, `fMARCOS`, `fPIV`, `fRAFSI`, `fRIM` — see the [methods table](#modules-and-functionalities) below for details and references.

### New subjective weighting methods (6)

`fAHP`, `fBWM`, `fFUCOM`, `fLMAW`, `fRANCOM`, `fSWARA`, available under `pyfdm.weights.subjective`. These complement the existing objective weighting methods (`equal_weights`, `shannon_entropy_weights`, `standard_deviation_weights`, `variance_weights`), which remain available from `pyfdm.weights` for backwards compatibility and are now also organised under `pyfdm.weights.objective`.

### Expert data collection and export

`pyfdm.expert.ExpertCollector` collects integer ratings from multiple experts using a configurable linguistic scale (`SCALE_1_5`, `SCALE_1_7`, `SCALE_1_9`, or a custom scale) and converts them to TFN decision matrices. `pyfdm.expert.FuzzyExporter` exports collected data to CSV, JSON, or Excel.

### Group aggregation

`pyfdm.group.aggregate()` combines TFN matrices from multiple experts using arithmetic mean, geometric mean, weighted average, or OWA (Ordered Weighted Average).

### Intermediate-result inspection (`logger`)

Every MCDA method and subjective weighting method now accepts a `logger=` keyword argument in `__init__`. Pass a `pyfdm.step_logger.StepLogger` instance to capture every intermediate matrix and vector the method computes (normalized matrix, weighted matrix, distances, etc.) and render it as a console table, JSON file, CSV file, or Excel workbook — or any combination. See the usage example below.

### Standalone ranking utility

`pyfdm.utils.rank_alternatives()` computes a ranking directly from any 1-D array of scores, independent of running an MCDA method, with five tie-breaking strategies (`average`, `min`, `max`, `dense`, `ordinal`) and full input validation. Useful for ranking externally computed scores (e.g. expert ratings, model outputs).

---

# Modules and functionalities

### Fuzzy MCDA methods

| Abbreviation | Full name                                                                                                  | Reference      |
| ------------ | ---------------------------------------------------------------------------------------------------------- | -------------- |
| fARAS        | fuzzy Additive Ratio ASsessment                                                                            | [[1]](#ref1)   |
| fAROMAN      | fuzzy Alternative Ranking Order Method Accounting for two-step Normalization                               | [[33]](#ref33) |
| fCOCOSO      | fuzzy Combined Compromise Solution                                                                         | [[32]](#ref32) |
| fCODAS       | fuzzy COmbinative Distance-based ASsessment                                                                | [[2]](#ref2)   |
| fCOPRAS      | fuzzy COmplex PRoportional ASsessment                                                                      | [[3]](#ref3)   |
| fEDAS        | fuzzy Evaluation based on Distance from Average Solution                                                   | [[4]](#ref4)   |
| fERVD        | fuzzy Election based on Relative Value Distances                                                           | [[34]](#ref34) |
| fMABAC       | fuzzy Multi-Attributive Border Approximation area Comparison                                               | [[5]](#ref5)   |
| fMAIRCA      | fuzzy MultiAttributive Ideal-Real Comparative Analysis                                                     | [[6]](#ref6)   |
| fMARCOS      | fuzzy Measurement of Alternatives and Ranking according to Compromise Solution                             | [[35]](#ref35) |
| fMOORA       | fuzzy Multi-Objective Optimization Method by Ratio Analysis                                                | [[7]](#ref7)   |
| fOCRA        | fuzzy Operational Competitiveness Ratings                                                                  | [[8]](#ref8)   |
| fPIV         | fuzzy Proximity Indexed Value                                                                              | [[36]](#ref36) |
| fRAFSI       | fuzzy Ranking of Alternatives through Functional mapping of criterion Sub-Intervals into a Single Interval | [[37]](#ref37) |
| fRIM         | fuzzy Reference Ideal Method                                                                               | [[38]](#ref38) |
| fSPOTIS      | fuzzy Stable Preference Ordering Towards Ideal Solution                                                    | [[25]](#ref25) |
| fTOPSIS      | fuzzy Technique for the Order of Prioritisation by Similarity to Ideal Solution                            | [[9]](#ref9)   |
| fVIKOR       | fuzzy VIseKriterijumska Optimizacija I Kompromisno Resenje                                                 | [[10]](#ref10) |
| fWASPAS      | fuzzy Weighted Aggregated Sum Product Assessment                                                           | [[26]](#ref26) |
| fWPM         | fuzzy Weighted Product Model                                                                               | [[27]](#ref27) |
| fWSM         | fuzzy Weighted Sum Model                                                                                   | [[27]](#ref27) |

### Weighting methods — objective

| Name                       | Reference      |
| -------------------------- | -------------- |
| Equal weights              | [[11]](#ref11) |
| Shannon entropy weights    | [[12]](#ref12) |
| Standard deviation weights | [[13]](#ref13) |
| Variance weights           | [[14]](#ref14) |

### Weighting methods — subjective

| Abbreviation | Full name                                         | Reference      |
| ------------ | ------------------------------------------------- | -------------- |
| fAHP         | fuzzy Analytic Hierarchy Process                  | [[39]](#ref39) |
| fBWM         | fuzzy Best-Worst Method                           | [[40]](#ref40) |
| fFUCOM       | fuzzy Full Consistency Method                     | [[41]](#ref41) |
| fLMAW        | fuzzy Logarithm Methodology of Additive Weights   | [[42]](#ref42) |
| fRANCOM      | fuzzy RANking COMparison                          | [[43]](#ref43) |
| fSWARA       | fuzzy Step-wise Weight Assessment Ratio Analysis. | [[44]](#ref44) |

### Normalization methods

| Name                  | Reference                    |
| --------------------- | ---------------------------- |
| COCOSO Normalization  | [[32]](#ref32)               |
| Linear Normalization  | [[15]](#ref15)               |
| Max Normalization     | [[2]](#ref2)                 |
| Min-Max Normalization | [[5]](#ref5)                 |
| SAW Normalization     | [[3]](#ref3), [[24]](#ref24) |
| Sum Normalization     | [[1]](#ref1)                 |
| Sqrt Normalization    | [[31]](#ref31)               |
| Vector Normalization  | [[7]](#ref7)                 |
| WASPAS Normalization  | [[26]](#ref26)               |

### Defuzzification methods

| Name                                | Reference                     |
| ----------------------------------- | ----------------------------- |
| Bisector defuzzification            | [[29]](#ref29)                |
| Graded mean average defuzzification | [[4]](#ref4)                  |
| Height defuzzification              | [[29]](#ref29)                |
| Largest of Maximum defuzzification  | [[29]](#ref29)                |
| Mean defuzzification                | [[16]](#ref16) [[17]](#ref17) |
| Mean area defuzzification           | [[15]](#ref15)                |
| Smallest of Maximum defuzzification | [[29]](#ref29)                |
| Weighted mean defuzzification       | [[10]](#ref10)                |

### Distance measures

| Name                        | Reference      |
| --------------------------- | -------------- |
| Canberra distance           | [[30]](#ref30) |
| Chebyshev distance          | [[30]](#ref30) |
| Euclidean distance          | [[18]](#ref18) |
| Hamming distance            | [[19]](#ref19) |
| Mahdavi distance            | [[18]](#ref18) |
| L-R distance                | [[19]](#ref19) |
| Tran Duckstein distance     | [[19]](#ref19) |
| Vertex distance             | [[15]](#ref15) |
| Weighted Euclidean distance | [[15]](#ref15) |
| Weighted Hamming distance   | [[15]](#ref15) |

### Correlation coefficients

| Name                                      | Reference      |
| ----------------------------------------- | -------------- |
| Pearson correlation coefficient           | [[21]](#ref21) |
| Spearman correlation coefficient          | [[20]](#ref20) |
| Weighted Spearman correlation coefficient | [[22]](#ref22) |
| WS Rank Similarity coefficient            | [[23]](#ref23) |

### Triangular Fuzzy Number [[28]](#ref28)

| Functionality name       |
| ------------------------ |
| Addition                 |
| Subtractions             |
| Multiplication           |
| Division                 |
| Absolute value           |
| Equality                 |
| Less equal comparison    |
| Greater equal comparison |
| Round value              |
| Membership function      |
| Centroid                 |
| Core                     |
| Inclusion                |
| S-norm operator          |
| T-norm operator          |

### Graphs

| Functionality name   |
| -------------------- |
| Multiple TFNs plot   |
| Single TFN plot      |
| S-norm operator plot |
| T-norm operator plot |
| TFN criteria plot    |
| TFN membership plot  |

### Helper / utility functions

| Function                | Module        | Description                                                               |
| ----------------------- | ------------- | ------------------------------------------------------------------------- |
| `rank`                  | `pyfdm.utils` | Legacy ranking helper (delegates to `rank_alternatives`).                 |
| `rank_alternatives`     | `pyfdm.utils` | Standalone ranking from any score vector, with 5 tie-breaking strategies. |
| `generate_fuzzy_matrix` | `pyfdm.utils` | Generates a random TFN decision matrix.                                   |
| `normalize_weights`     | `pyfdm.utils` | Normalizes fuzzy criteria weights to `[0, 1]`.                            |
| `defuzzify_matrix`      | `pyfdm.utils` | Defuzzifies an entire TFN decision matrix.                                |

### Expert data collection and export

| Component                             | Module                | Description                                                            |
| ------------------------------------- | --------------------- | ---------------------------------------------------------------------- |
| `ExpertCollector`                     | `pyfdm.expert`        | Collects per-expert integer ratings and converts them to TFN matrices. |
| `FuzzyExporter`                       | `pyfdm.expert`        | Exports collected TFN data to CSV, JSON, or Excel.                     |
| `SCALE_1_5`, `SCALE_1_7`, `SCALE_1_9` | `pyfdm.expert.scales` | Built-in linguistic-to-TFN rating scales.                              |

### Group aggregation

| Function    | Module        | Description                                                                                   |
| ----------- | ------------- | --------------------------------------------------------------------------------------------- |
| `aggregate` | `pyfdm.group` | Combines multiple expert TFN matrices (`'geometric'`, `'arithmetic'`, `'weighted'`, `'owa'`). |

# Usage examples

More usage examples are available in [Jupyter examples](https://github.com/jwieckowski/pyfdm/blob/main/examples/examples.ipynb).

## Basic MCDA ranking

```python
from pyfdm.methods import fARAS
import numpy as np

matrix = np.array([
    [[5, 7, 9], [5, 7, 9], [7, 9, 9]],
    [[1, 3, 5], [3, 5, 7], [3, 5, 7]],
    [[1, 1, 3], [1, 3, 5], [1, 3, 5]],
    [[7, 9, 9], [7, 9, 9], [7, 9, 9]]
])

weights = np.array([[5, 7, 9], [7, 9, 9], [3, 5, 7]])
types = np.array([1, -1, 1])

f_aras = fARAS()
pref = f_aras(matrix, weights, types)

print(f'Fuzzy ARAS preferences: {pref}')
print(f'Fuzzy ARAS ranking: {f_aras.rank()}')
```

Output:

```
Fuzzy ARAS preferences: [1.011 0.854 1.312 0.993]
Fuzzy ARAS ranking: [2. 4. 1. 3.]
```

## Inspecting intermediate computation steps

Every MCDA method accepts `logger=` to capture all intermediate matrices it computes (normalized matrix, weighted matrix, distances, etc.).

```python
from pyfdm.methods import fMARCOS
from pyfdm.step_logger import StepLogger
import numpy as np

matrix = np.array([
    [[5, 7, 9], [5, 7, 9], [7, 9, 9]],
    [[1, 3, 5], [3, 5, 7], [3, 5, 7]],
    [[1, 1, 3], [1, 3, 5], [1, 3, 5]],
    [[7, 9, 9], [7, 9, 9], [7, 9, 9]]
], dtype=float)
weights = np.array([0.4, 0.35, 0.25])
types = np.array([1, -1, 1])

# Print to console
logger = StepLogger(output='console')
method = fMARCOS(logger=logger)
pref = method(matrix, weights, types)

# Or export to selected format
logger = StepLogger(output='json', path='results/marcos_steps')
method = fMARCOS(logger=logger)
pref = method(matrix, weights, types)
# -> results/marcos_steps.json,
```

Available `output` formats: `'console'`, `'json'`, `'csv'`, `'excel'` (requires `pip install pyfdm[excel]`). Pass a single string or a list to combine formats.

## Subjective weighting (fAHP)

```python
from pyfdm.weights.subjective import fAHP
import numpy as np

# Pairwise TFN comparison matrix (criterion i vs criterion j)
comparison_matrix = np.array([
    [(1, 1, 1),       (2, 3, 4),       (4, 5, 6)],
    [(1/4, 1/3, 1/2), (1, 1, 1),       (2, 3, 4)],
    [(1/6, 1/5, 1/4), (1/4, 1/3, 1/2), (1, 1, 1)],
], dtype=float)

ahp = fAHP()
weights = ahp(comparison_matrix)   # shape (3, 3), TFN weights
```

## Collecting expert judgements and exporting

```python
from pyfdm.expert import ExpertCollector, FuzzyExporter
from pyfdm.expert.scales import SCALE_1_9

collector = ExpertCollector(n_alternatives=4, n_criteria=3, scale=SCALE_1_9)
collector.add_expert_matrix(0, [[9, 3, 7], [5, 9, 3], [3, 5, 9], [7, 3, 5]])
collector.add_expert_matrix(1, [[7, 5, 9], [9, 3, 5], [5, 7, 3], [3, 9, 7]])

exporter = FuzzyExporter(collector)
exporter.to_csv('expert_ratings.csv')
exporter.to_json('expert_ratings.json')
exporter.to_excel('expert_ratings.xlsx')   # requires openpyxl
```

## Group aggregation of expert matrices

```python
from pyfdm.group import aggregate

matrices = collector.get_tfn_matrices()           # list of (m, n, 3) arrays
agg_matrix = aggregate(matrices, method='geometric')

# or weighted by expert importance
agg_matrix = aggregate(matrices, method='weighted', weights=[0.6, 0.4])
```

## Standalone ranking of external scores

Rank any 1-D array of scores directly, without running an MCDA method:

```python
from pyfdm.methods import rank_alternatives

scores = [72.5, 88.0, 88.0, 60.1, 95.3]
ranking = rank_alternatives(scores, descending=True, method='average')
print(ranking)   # [3. 2.5 2.5 5. 1.]
```

---

# Citations

If you use **pyFDM** in your research to perform Fuzzy Multi-Criteria Decision Analysis (Fuzzy MCDA), please cite the appropriate publication corresponding to the version of the library used.

## Initial version

**APA:**

> Więckowski, J., Kizielewicz, B., & Sałabun, W. (2022). pyFDM: A Python library for uncertainty decision analysis methods. _SoftwareX, 20_, 101271.

[https://doi.org/10.1016/j.softx.2022.101271](https://doi.org/10.1016/j.softx.2022.101271)

**BibTeX:**

```bibtex
@article{wikeckowski2022pyfdm,
  title={pyFDM: A Python library for uncertainty decision analysis methods},
  author={Wi{\k{e}}ckowski, Jakub and Kizielewicz, Bart{\l}omiej and Sa{\l}abun, Wojciech},
  journal={SoftwareX},
  volume={20},
  pages={101271},
  year={2022},
  publisher={Elsevier}
}
```

## Version 1.1

**APA:**

> Więckowski, J., & Sałabun, W. (2024). Version [1.1]-[pyFDM: A python library for uncertainty decision analysis methods]. SoftwareX, 25, 101607.

[https://doi.org/10.1016/j.softx.2023.101607](https://doi.org/10.1016/j.softx.2023.101607)

**BibTeX:**

```bibtex
@article{wikeckowski2024version,
  title={Version [1.1]-[pyFDM: A python library for uncertainty decision analysis methods]},
  author={Wi{\k{e}}ckowski, Jakub and Sa{\l}abun, Wojciech},
  journal={SoftwareX},
  volume={25},
  pages={101607},
  year={2024},
  publisher={Elsevier}
}
```

# References

<a name="ref1">**[1]**</a> Fu, Y. K., Wu, C. J., & Liao, C. N. (2021). Selection of in-flight duty-free product suppliers using a combination fuzzy AHP, fuzzy ARAS, and MSGP methods. Mathematical Problems in Engineering, 2021.

<a name="ref2">**[2]**</a> Panchal, D., Chatterjee, P., Shukla, R. K., Choudhury, T., & Tamosaitiene, J. (2017). Integrated Fuzzy AHP-Codas Framework for Maintenance Decision in Urea Fertilizer Industry. Economic Computation & Economic Cybernetics Studies & Research, 51(3).

<a name="ref3">**[3]**</a> Narang, M., Joshi, M. C., & Pal, A. K. (2021). A hybrid fuzzy COPRAS-base-criterion method for multi-criteria decision making. Soft Computing, 25(13), 8391-8399.

<a name="ref4">**[4]**</a> Zindani, D., Maity, S. R., & Bhowmik, S. (2019). Fuzzy-EDAS (evaluation based on distance from average solution) for material selection problems. In Advances in Computational Methods in Manufacturing (pp. 755-771). Springer, Singapore.

<a name="ref5">**[5]**</a> Bozanic, D., Tešić, D., & Milićević, J. (2018). A hybrid fuzzy AHP-MABAC model: Application in the Serbian Army–The selection of the location for deep wading as a technique of crossing the river by tanks. Decision Making: Applications in Management and Engineering, 1(1), 143-164.

<a name="ref6">**[6]**</a> Boral, S., Howard, I., Chaturvedi, S. K., McKee, K., & Naikan, V. N. A. (2020). An integrated approach for fuzzy failure modes and effects analysis using fuzzy AHP and fuzzy MAIRCA. Engineering Failure Analysis, 108, 104195.

<a name="ref7">**[7]**</a> Karande, P., & Chakraborty, S. (2012). A Fuzzy-MOORA approach for ERP system selection. Decision Science Letters, 1(1), 11-21.

<a name="ref8">**[8]**</a> ULUTAŞ, A. (2019). Supplier selection by using a fuzzy integrated model for a textile company. Engineering Economics, 30(5), 579-590.

<a name="ref9">**[9]**</a> Chen, C. T. (2000). Extensions of the TOPSIS for group decision-making under fuzzy environment. Fuzzy sets and systems, 114(1), 1-9.

<a name="ref10">**[10]**</a> Opricovic, S. (2007). A fuzzy compromise solution for multicriteria problems. International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, 15(03), 363-380.

<a name="ref11">**[11]**</a> Iskander, M. G. (2002). Comparison of fuzzy numbers using possibility programming: comments and new concepts. Computers & Mathematics with Applications, 43(6-7), 833-840.

<a name="ref12">**[12]**</a> Kacprzak, D. (2017). Objective weights based on ordered fuzzy numbers for fuzzy multiple criteria decision-making methods. Entropy, 19(7), 373.

<a name="ref13">**[13]**</a> Wang, Y. M., & Luo, Y. (2010). Integration of correlations with standard deviations for determining attribute weights in multiple attribute decision making. Mathematical and Computer Modelling, 51(1-2), 1-12.

<a name="ref14">**[14]**</a> Bikmukhamedov, R., Yeryomin, Y., & Seitz, J. (2016, July). Evaluation of MCDA-based handover algorithms for mobile networks. In 2016 Eighth International Conference on Ubiquitous and Future Networks (ICUFN) (pp. 810-815). IEEE.

<a name="ref15">**[15]**</a> Roszkowska, E., & Wachowicz, T. (2015). Application of fuzzy TOPSIS to scoring the negotiation offers in ill-structured negotiation problems. European Journal of Operational Research, 242(3), 920-932.

<a name="ref16">**[16]**</a> Yılmaz, M., & Atan, T. (2021). Hospital site selection using fuzzy EDAS method: case study application for districts of Istanbul. Journal of Intelligent & Fuzzy Systems, (Preprint), 1-12.

<a name="ref17">**[17]**</a> Zolfani, S. H., Görçün, Ö. F., & Küçükönder, H. (2021). Evaluating logistics villages in Turkey using hybrid improved fuzzy SWARA (IMF SWARA) and fuzzy MABAC techniques. Technological and Economic Development of Economy, 27(6), 1582-1612.

<a name="ref18">**[18]**</a> Wang, H., Lu, X., Du, Y., Zhang, C., Sadiq, R., & Deng, Y. (2017). Fault tree analysis based on TOPSIS and triangular fuzzy number. International journal of system assurance engineering and management, 8(4), 2064-2070.

<a name="ref19">**[19]**</a> Talukdar, P., & Dutta, P. A Comparative Study of TOPSIS Method via Different Distance Measure.

<a name="ref20">**[20]**</a> Spearman, C. (1910). Correlation calculated from faulty data. British Journal of Psychology, 1904‐1920, 3(3), 271-295.

<a name="ref21">**[21]**</a> Pearson, K. (1895). VII. Note on regression and inheritance in the case of two parents. proceedings of the royal society of London, 58(347-352), 240-242.

<a name="ref22">**[22]**</a> Dancelli, L., Manisera, M., & Vezzoli, M. (2013). On two classes of Weighted Rank Correlation measures deriving from the Spearman's rho. In Statistical Models for Data Analysis (pp. 107-114). Springer, Heidelberg.

<a name="ref23">**[23]**</a> Sałabun, W., & Urbaniak, K. (2020, June). A new coefficient of rankings similarity in decision-making problems. In International Conference on Computational Science (pp. 632-645). Springer, Cham.

<a name="ref24">**[24]**</a> Saifullah, S. (2021). Fuzzy-AHP approach using Normalized Decision Matrix on Tourism Trend Ranking based-on Social Media. arXiv preprint arXiv:2102.04222.

<a name="ref25">**[25]**</a> Shekhovtsov, A., Paradowski, B., Więckowski, J., Kizielewicz, B., & Sałabun, W. (2022, December). Extension of the SPOTIS method for the rank reversal free decision-making under fuzzy environment. In 2022 IEEE 61st Conference on Decision and Control (CDC) (pp. 5595-5600). IEEE.

<a name="ref26">**[26]**</a> Turskis, Z., Zavadskas, E. K., Antuchevičienė, J., & Kosareva, N. (2015). A hybrid model based on fuzzy AHP and fuzzy WASPAS for construction site selection.

<a name="ref27">**[27]**</a> Triantaphyllou, E., & Lin, C. T. (1996). Development and evaluation of five fuzzy multiattribute decision-making methods. international Journal of Approximate reasoning, 14(4), 281-310.

<a name="ref28">**[28]**</a> Sudha, T., & Jayalalitha, G. (2020, July). Fuzzy triangular numbers in-Sierpinski triangle and right angle triangle. In Journal of Physics: Conference Series (Vol. 1597, No. 1, p. 012022). IOP Publishing.

<a name="ref29">**[29]**</a> Berkachy, R., & Donzé, L. (2016). Linguistic questionnaire evaluation: an application of the signed distance defuzzification method on different fuzzy numbers. The impact on the skewness of the output distributions. International Journal of Fuzzy Systems and Advanced Applications, 3, 12-19.

<a name="ref30">**[30]**</a> Rodrigues, É. O. (2018). Combining Minkowski and Chebyshev: New distance proposal and survey of distance metrics using k-nearest neighbours classifier. Pattern Recognition Letters, 110, 66-71.

<a name="ref31">**[31]**</a> Kizielewicz, B., & Bączkiewicz, A. (2021). Comparison of Fuzzy TOPSIS, Fuzzy VIKOR, Fuzzy WASPAS and Fuzzy MMOORA methods in the housing selection problem. Procedia Computer Science, 192, 4578-4591.

<a name="ref32">**[32]**</a> Ulutaş, A., Popovic, G., Radanov, P., Stanujkic, D., & Karabasevic, D. (2021). A new hybrid fuzzy PSI-PIPRECIA-CoCoSo MCDM based approach to solving the transportation company selection problem. Technological and Economic Development of Economy, 27(5), 1227-1249.

<a name="ref33">**[33]**</a> Čubranić-Dobrodolac, M., Jovčić, S., Bošković, S., & Babić, D. (2023). A decision-making model for professional drivers selection: A hybridized fuzzy-AROMAN-Fuller approach. Mathematics, 11(13), 2831.

<a name="ref34">**[34]**</a> Shojaeimehr, S., & Rahmani, D. (2022). Risk management of photovoltaic power plants using a novel fuzzy multi-criteria decision-making method based on prospect theory: A sustainable development approach. Energy Conversion and Management: X, 16, 100293.

<a name="ref35">**[35]**</a> Stanković, M., Stević, Ž., Das, D. K., Subotić, M., & Pamučar, D. (2020). A new fuzzy MARCOS method for road traffic risk analysis. Mathematics, 8(3), 457.

<a name="ref36">**[36]**</a> Seraj, M., Yahya, S. M., Badruddin, I. A., Anqi, A. E., Asjad, M., & Khan, Z. A. (2019). Multi-response optimization of nanofluid-based IC engine cooling system using fuzzy method. Processes, 8(1), 30.

<a name="ref37">**[37]**</a> Božanić, D., Milić, A., Tešić, D., Salabun, W., & Pamučar, D. (2021). D numbers–FUCOM–fuzzy RAFSI model for selecting the group of construction machines for enabling mobility. Facta Universitatis, Series: Mechanical Engineering, 19(3), 447-471.

<a name="ref38">**[38]**</a> Cables, E., Lamata, M. T., & Verdegay, J. L. (2017). FRIM—fuzzy reference ideal method in multicriteria decision making. In Soft computing applications for group decision-making and consensus modeling (pp. 305-317). Cham: Springer International Publishing.

<a name="ref39">**[39]**</a> Sun, C. C. (2010). A performance evaluation model by integrating fuzzy AHP and fuzzy TOPSIS methods. Expert systems with applications, 37(12), 7745-7754.

<a name="ref40">**[40]**</a> Guo, S., & Zhao, H. (2017). Fuzzy best-worst multi-criteria decision-making method and its applications. Knowledge-Based Systems, 121, 23-31.

<a name="ref41">**[41]**</a> Pamucar, D., & Ecer, F. (2020). Prioritizing the weights of the evaluation criteria under fuzziness: the fuzzy full consistency method - FUCOM-F. Facta Universitatis, Series: Mechanical Engineering, 18(3), 419-437.

<a name="ref42">**[42]**</a> Božanić, D., Pamučar, D., Milić, A., Marinković, D., & Komazec, N. (2022). Modification of the logarithm methodology of additive weights (LMAW) by a triangular fuzzy number and its application in multi-criteria decision making. Axioms, 11(3), 89.

<a name="ref43">**[43]**</a> Więckowski, J., Kizielewicz, B., & Sałabun, W. (2025). Fuzzy RANCOM: a novel approach for modeling uncertainty in decision-making processes. Information sciences, 694, 121716.

<a name="ref44">**[44]**</a> Mehdiabadi, A., Sadeghi, A., Karbassi Yazdi, A., & Tan, Y. (2025). Sustainability Service Chain Capabilities in the Oil and Gas Industry: A Fuzzy Hybrid Approach SWARA-MABAC. Spectrum of Operational Research, 2(1), 114-134.
