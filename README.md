# pyfdm

Python 3 package with Fuzzy Decision Making (PyFDM) methods based on Triangular Fuzzy Numbers (TFN)

---

# Installation

The package can be download using pip:

```Bash
pip install pyfdm
```

# Testing

The modules performance can be verified with pytest library

```Bash
pip install pytest
pytest tests
```

---

# Modules and functionalities

- Fuzzy MCDA methods:

| Abbreviation | Full name                                                                 | Reference |
| ------------ | ------------------------------------------------------------------------- | --------- |
| ARAS         | Additive Ratio ASsessment                                                 | [1]       |
| CODAS        | COmbinative Distance-based ASsessment                                     | [2]       |
| COPRAS       | COmplex PRoportional ASsessment                                           | [3]       |
| EDAS         | Evaluation based on Distance from Average Solution                        | [4]       |
| MABAC        | Multi-Attributive Border Approximation area Comparison                    | [5]       |
| MAIRCA       | MultiAttributive Ideal-Real Comparative Analysis                          | [6]       |
| MOORA        | Multi-Objective Optimization Method by Ratio Analysis                     | [7]       |
| OCRA         | Operational Competitiveness Ratings                                       | [8]       |
| TOPSIS       | Technique for the Order of Prioritisation by Similarity to Ideal Solution | [9]       |
| VIKOR        | VIseKriterijumska Optimizacija I Kompromisno Resenje                      | [10]      |

- Weighting methods:

| Name                       | Reference |
| -------------------------- | :-------: |
| Equal weights              |   [11]    |
| Shannon entropy weights    |   [12]    |
| Standard deviation weights |   [13]    |
| Variance weights           |   [14]    |

- Normalization methods:

| Name                  | Reference |
| --------------------- | :-------: |
| Sum Normalization     |    [1]    |
| Max Normalization     |    [2]    |
| Linear Normalization  |   [15]    |
| Min-Max Normalization |    [5]    |
| Vector Normalization  |    [7]    |
| SAW Normalization     |   [3,24]  |

- Defuzzification methods:

| Name                                | Reference |
| ----------------------------------- | :-------: |
| Mean defuzzification                |  [16,17]  |
| Mean area defuzzification           |   [15]    |
| Graded mean average defuzzification |    [4]    |
| Weighted mean defuzzification       |   [10]    |

- Distance measures:

| Name                        | Reference |
| --------------------------- | :-------: |
| Euclidean distance          |   [18]    |
| Weighted Euclidean distance |   [15]    |
| Hamming distance            |   [19]    |
| Weighted Hamming distance   |   [15]    |
| Vertex distance             |   [15]    |
| Tran Duckstein distance     |   [19]    |
| L-R distance                |   [19]    |
| Mahdavi distance            |   [18]    |

- Correlation coefficients:

| Name                                      | Reference |
| ----------------------------------------- | :-------: |
| Spearman correlation coefficient          |   [20]    |
| Pearson correlation coefficient           |   [21]    |
| Weighted Spearman correlation coefficient |   [22]    |
| WS Rank Similarity coefficient            |   [23]    |

- Helpers methods
  - rank
  - generarte_fuzzy_matrix

# Usage example

Below the sample example of the package functionalities is presented.
More usage examples of available methods are presented in [Jupyter examples](https://github.com/jwieckowski/pyfdm).

```python
from pyfdm.methods import fARAS
from pyfdm.helpers import rank
import numpy as np

if __name__ == '__main__':
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
    print(f'Fuzzy ARAS ranking: {rank(pref)}')
```

Output:

```bash
Fuzzy ARAS preferences: 1.011 0.854 1.312 0.993
Fuzzy ARAS ranking: 2 4 1 3
```


### References

[1] Fu, Y. K., Wu, C. J., & Liao, C. N. (2021). Selection of in-flight duty-free product suppliers using a combination fuzzy AHP, fuzzy ARAS, and MSGP methods. Mathematical Problems in Engineering, 2021.

[2] Panchal, D., Chatterjee, P., Shukla, R. K., Choudhury, T., & Tamosaitiene, J. (2017). Integrated Fuzzy AHP-Codas Framework for Maintenance Decision in Urea Fertilizer Industry. Economic Computation & Economic Cybernetics Studies & Research, 51(3).

[3] Narang, M., Joshi, M. C., & Pal, A. K. (2021). A hybrid fuzzy COPRAS-base-criterion method for multi-criteria decision making. Soft Computing, 25(13), 8391-8399.

[4] Zindani, D., Maity, S. R., & Bhowmik, S. (2019). Fuzzy-EDAS (evaluation based on distance from average solution) for material selection problems. In Advances in Computational Methods in Manufacturing (pp. 755-771). Springer, Singapore.

[5] Bozanic, D., Tešić, D., & Milićević, J. (2018). A hybrid fuzzy AHP-MABAC model: Application in the Serbian Army–The selection of the location for deep wading as a technique of crossing the river by tanks. Decision Making: Applications in Management and Engineering, 1(1), 143-164.

[6] Boral, S., Howard, I., Chaturvedi, S. K., McKee, K., & Naikan, V. N. A. (2020). An integrated approach for fuzzy failure modes and effects analysis using fuzzy AHP and fuzzy MAIRCA. Engineering Failure Analysis, 108, 104195.

[7] Karande, P., & Chakraborty, S. (2012). A Fuzzy-MOORA approach for ERP system selection. Decision Science Letters, 1(1), 11-21.

[8] ULUTAŞ, A. (2019). Supplier selection by using a fuzzy integrated model for a textile company. Engineering Economics, 30(5), 579-590.

[9] Chen, C. T. (2000). Extensions of the TOPSIS for group decision-making under fuzzy environment. Fuzzy sets and systems, 114(1), 1-9.

[10] Opricovic, S. (2007). A fuzzy compromise solution for multicriteria problems. International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems, 15(03), 363-380.

[11] Iskander, M. G. (2002). Comparison of fuzzy numbers using possibility programming: comments and new concepts. Computers & Mathematics with Applications, 43(6-7), 833-840.

[12] Kacprzak, D. (2017). Objective weights based on ordered fuzzy numbers for fuzzy multiple criteria decision-making methods. Entropy, 19(7), 373.

[13] Wang, Y. M., & Luo, Y. (2010). Integration of correlations with standard deviations for determining attribute weights in multiple attribute decision making. Mathematical and Computer Modelling, 51(1-2), 1-12.

[14] Bikmukhamedov, R., Yeryomin, Y., & Seitz, J. (2016, July). Evaluation of MCDA-based handover algorithms for mobile networks. In 2016 Eighth International Conference on Ubiquitous and Future Networks (ICUFN) (pp. 810-815). IEEE.

[15] Roszkowska, E., & Wachowicz, T. (2015). Application of fuzzy TOPSIS to scoring the negotiation offers in ill-structured negotiation problems. European Journal of Operational Research, 242(3), 920-932.

[16] Yılmaz, M., & Atan, T. (2021). Hospital site selection using fuzzy EDAS method: case study application for districts of Istanbul. Journal of Intelligent & Fuzzy Systems, (Preprint), 1-12.

[17] Zolfani, S. H., Görçün, Ö. F., & Küçükönder, H. (2021). Evaluating logistics villages in Turkey using hybrid improved fuzzy SWARA (IMF SWARA) and fuzzy MABAC techniques. Technological and Economic Development of Economy, 27(6), 1582-1612.

[18] Wang, H., Lu, X., Du, Y., Zhang, C., Sadiq, R., & Deng, Y. (2017). Fault tree analysis based on TOPSIS and triangular fuzzy number. International journal of system assurance engineering and management, 8(4), 2064-2070.

[19] Talukdar, P., & Dutta, P. A Comparative Study of TOPSIS Method via Different Distance Measure.

[20] Spearman, C. (1910). Correlation calculated from faulty data. British Journal of Psychology, 1904‐1920, 3(3), 271-295.

[21] Pearson, K. (1895). VII. Note on regression and inheritance in the case of two parents. proceedings of the royal society of London, 58(347-352), 240-242.

[22] Dancelli, L., Manisera, M., & Vezzoli, M. (2013). On two classes of Weighted Rank Correlation measures deriving from the Spearman’s ρ. In Statistical Models for Data Analysis (pp. 107-114). Springer, Heidelberg.

[23] Sałabun, W., & Urbaniak, K. (2020, June). A new coefficient of rankings similarity in decision-making problems. In International Conference on Computational Science (pp. 632-645). Springer, Cham.

[24] Saifullah, S. (2021). Fuzzy-AHP approach using Normalized Decision Matrix on Tourism Trend Ranking based-on Social Media. arXiv preprint arXiv:2102.04222.