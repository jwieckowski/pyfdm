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

# Citations

If you are using this library in your research work to calculate results with Fuzzy MCDA approach, cite with APA format :

"[Więckowski, J., Kizielewicz, B., & Sałabun, W. (2022). pyFDM: A Python library for uncertainty decision analysis methods. SoftwareX, 20, 101271.](https://www.sciencedirect.com/science/article/pii/S2352711022001893)"

or with BibTex :

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

# Modules and functionalities

- Fuzzy MCDA methods:

| Abbreviation | Full name                                                                 | Reference      |
| ------------ | ------------------------------------------------------------------------- | -------------- |
| ARAS         | Additive Ratio ASsessment                                                 | [[1]](#ref1)   |
| COCOSO       | Combined Compromise Solution                                              | [[32]](#ref32) |
| CODAS        | COmbinative Distance-based ASsessment                                     | [[2]](#ref2)   |
| COPRAS       | COmplex PRoportional ASsessment                                           | [[3]](#ref3)   |
| EDAS         | Evaluation based on Distance from Average Solution                        | [[4]](#ref4)   |
| MABAC        | Multi-Attributive Border Approximation area Comparison                    | [[5]](#ref5)   |
| MAIRCA       | MultiAttributive Ideal-Real Comparative Analysis                          | [[6]](#ref6)   |
| MOORA        | Multi-Objective Optimization Method by Ratio Analysis                     | [[7]](#ref7)   |
| OCRA         | Operational Competitiveness Ratings                                       | [[8]](#ref8)   |
| SPOTIS       | Stable Preference Ordering Towards Ideal Solution                         | [[25]](#ref25) |
| TOPSIS       | Technique for the Order of Prioritisation by Similarity to Ideal Solution | [[9]](#ref9)   |
| VIKOR        | VIseKriterijumska Optimizacija I Kompromisno Resenje                      | [[10]](#ref10) |
| WASPAS       | Weighted Aggregated Sum Product Assessment                                | [[26]](#ref26) |
| WPM          | Weighted Product Model                                                    | [[27]](#ref26) |
| WSM          | Weighted Sum Model                                                        | [[27]](#ref26) |

- Weighting methods:

| Name                       |   Reference    |
| -------------------------- | :------------: |
| Equal weights              | [[11]](#ref11) |
| Shannon entropy weights    | [[12]](#ref12) |
| Standard deviation weights | [[13]](#ref13) |
| Variance weights           | [[14]](#ref14) |

- Normalization methods:

| Name                  |          Reference           |
| --------------------- | :--------------------------: |
| COCOSO Normalization  |        [[32]](#ref32)        |
| Linear Normalization  |        [[15]](#ref15)        |
| Max Normalization     |         [[2]](#ref2)         |
| Min-Max Normalization |         [[5]](#ref5)         |
| SAW Normalization     | [[3]](#ref3), [[24]](#ref24) |
| Sum Normalization     |         [[1]](#ref1)         |
| Sqrt Normalization    |        [[31]](#ref31)        |
| Vector Normalization  |         [[7]](#ref7)         |
| WASPAS Normalization  |        [[26]](#ref26)        |

- Defuzzification methods:

| Name                                |           Reference           |
| ----------------------------------- | :---------------------------: |
| Bisector defuzzification            |        [[29]](#ref29)         |
| Graded mean average defuzzification |         [[4]](#ref4)          |
| Height defuzzification              |        [[29]](#ref29)         |
| Largest of Maximum defuzzification  |        [[29]](#ref29)         |
| Mean defuzzification                | [[16]](#ref16) [[17]](#ref17) |
| Mean area defuzzification           |        [[15]](#ref15)         |
| Smallest of Maximum defuzzification |        [[29]](#ref29)         |
| Weighted mean defuzzification       |        [[10]](#ref10)         |

- Distance measures:

| Name                        |   Reference    |
| --------------------------- | :------------: |
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

- Correlation coefficients:

| Name                                      |   Reference    |
| ----------------------------------------- | :------------: |
| Pearson correlation coefficient           | [[21]](#ref21) |
| Spearman correlation coefficient          | [[20]](#ref20) |
| Weighted Spearman correlation coefficient | [[22]](#ref22) |
| WS Rank Similarity coefficient            | [[23]](#ref23) |

- Triangular Fuzzy Number [[28]](#ref28) :

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

- Graphs:

| Functionality name   |
| -------------------- |
| Multiple TFNs plot   |
| Single TFN plot      |
| S-norm operator plot |
| T-norm operator plot |
| TFN criteria plot    |
| TFN membership plot  |

- Helpers methods
  - rank
  - generate_fuzzy_matrix

# Usage example

Below the sample example of the package functionalities is presented.
More usage examples of available methods are presented in [Jupyter examples](https://github.com/jwieckowski/pyfdm/blob/main/examples/examples.ipynb).

```python
from pyfdm.methods import fARAS
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
    print(f'Fuzzy ARAS ranking: {f_aras.rank()}')
```

Output:

```bash
Fuzzy ARAS preferences: 1.011 0.854 1.312 0.993
Fuzzy ARAS ranking: 2 4 1 3
```

# References

<a name="ref1">**[1]**</a> Fu, Y. K., Wu, C. J., & Liao, C. N. (2021). Selection of in-flight duty-free product suppliers using a combination fuzzy AHP, fuzzy ARAS, and MSGP methods. Mathematical Problems in Engineering, 2021.

<a name="ref2">**[2]**</a>Panchal, D., Chatterjee, P., Shukla, R. K., Choudhury, T., & Tamosaitiene, J. (2017). Integrated Fuzzy AHP-Codas Framework for Maintenance Decision in Urea Fertilizer Industry. Economic Computation & Economic Cybernetics Studies & Research, 51(3).

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

<a name="ref22">**[22]**</a> Dancelli, L., Manisera, M., & Vezzoli, M. (2013). On two classes of Weighted Rank Correlation measures deriving from the Spearman’s ρ. In Statistical Models for Data Analysis (pp. 107-114). Springer, Heidelberg.

<a name="ref23">**[23]**</a> Sałabun, W., & Urbaniak, K. (2020, June). A new coefficient of rankings similarity in decision-making problems. In International Conference on Computational Science (pp. 632-645). Springer, Cham.

<a name="ref24">**[24]**</a> Saifullah, S. (2021). Fuzzy-AHP approach using Normalized Decision Matrix on Tourism Trend Ranking based-on Social Media. arXiv preprint arXiv:2102.04222.

<a name="ref25">**[25]**</a> Shekhovtsov, A., Paradowski, B., Więckowski, J., Kizielewicz, B., & Sałabun, W. (2022, December). Extension of the SPOTIS method for the rank reversal free decision-making under fuzzy environment. In 2022 IEEE 61st Conference on Decision and Control (CDC) (pp. 5595-5600). IEEE.

<a name="ref26">**[26]**</a> Turskis, Z., Zavadskas, E. K., Antuchevičienė, J., & Kosareva, N. (2015). A hybrid model based on fuzzy AHP and fuzzy WASPAS for construction site selection.

<a name="ref27">**[27]**</a> Triantaphyllou, E., & Lin, C. T. (1996). Development and evaluation of five fuzzy multiattribute decision-making methods. international Journal of Approximate reasoning, 14(4), 281-310.

<a name="ref28">**[28]**</a> Sudha, T., & Jayalalitha, G. (2020, July). Fuzzy triangular numbers in-Sierpinski triangle and right angle triangle. In Journal of Physics: Conference Series (Vol. 1597, No. 1, p. 012022). IOP Publishing.

<a name="ref29">**[29]**</a> Berkachy, R., & Donzé, L. (2016). Linguistic questionnaire evaluation: an application of the signed distance defuzzification method on different fuzzy numbers. The impact on the skewness of the output distributions. International Journal of Fuzzy Systems and Advanced Applications, 3, 12-19.

<a name="ref30">**[30]**</a> Rodrigues, É. O. (2018). Combining Minkowski and Chebyshev: New distance proposal and survey of distance metrics using k-nearest neighbours classifier. Pattern Recognition Letters, 110, 66-71.

<a name="ref31">**[31]**</a> Kizielewicz, B., \& Bączkiewicz, A. (2021). Comparison of Fuzzy TOPSIS, Fuzzy VIKOR, Fuzzy WASPAS and Fuzzy MMOORA methods in the housing selection problem. Procedia Computer Science, 192, 4578-4591.

<a name="ref32">**[32]**</a> Ulutaş, A., Popovic, G., Radanov, P., Stanujkic, D., & Karabasevic, D. (2021). A new hybrid fuzzy PSI-PIPRECIA-CoCoSo MCDM based approach to solving the transportation company selection problem. Technological and Economic Development of Economy, 27(5), 1227-1249.
