Below the examples of the different techniques implemented in the library are presented.

Imports
-------

.. code-block:: python
   :linenos:

   import numpy as np
   from tabulate import tabulate

   from pyfdm import methods
   from pyfdm.methods.fuzzy_sets import tfn
   from pyfdm import weights as f_weights
   from pyfdm import correlations as corrs
   from pyfdm.helpers import rank, generate_fuzzy_matrix

   import warnings

   np.set_printoptions(suppress=True, precision=3)

Input data
------------

To perform the multi-criteria evaluation, the decision matrix needs to be defined. It can be determined based on the real data, or created with the method provided in the library.

.. code-block:: python
   :linenos:

   # real data matrix
   real_matrix = np.array([
      [[5, 7, 9], [5, 7, 9], [7, 9, 9]],
      [[1, 3, 5], [3, 5, 7], [3, 5, 7]],
      [[1, 1, 3], [1, 3, 5], [1, 3, 5]],
      [[7, 9, 9], [7, 9, 9], [7, 9, 9]]
   ])

   # randomly generated matrix
   # 5 alternatives
   # 4 criteria
   # lower bound = 5
   # upper bound = 10
   random_matrix = generate_fuzzy_matrix(5, 4, 5, 10)
   print(random_matrix)

Normalization methods
--------------

Data normalization allows for comparing numbers with each other. It converts the range of values that they fit in range between 0 and 1. Below the usage examples of methods implemented in the library. Types parameter is responsible for the direction of the normalization. One columns' values could be more preferred is the values are lower (`-1`), other one could be more preferred if the values are greater (`1`).

.. code-block:: python
   :linenos:

   normalizations = {
      'Sum': tfn.normalizations.sum_normalization,
      'Max': tfn.normalizations.max_normalization,
      'Linear': tfn.normalizations.linear_normalization,
      'Minmax': tfn.normalizations.minmax_normalization,
      'Vector': tfn.normalizations.vector_normalization,
      'SAW': tfn.normalizations.saw_normalization
   }

   types = np.array([1, -1, 1])

   for name, norm in normalizations.items():
      nmatrix = norm(real_matrix, types)
      print(f'{name} \n {nmatrix[:2]}')

Distance metrics
---------

Distance is a measure that allows for indicating how far are two Triangular Fuzzy Numbers from each other. Different techniques have been developed to this end. The measures implemented in the library and their usage are presented below.

.. code-block:: python
   :linenos:
   
   distances = {
      'Euclidean': tfn.distances.euclidean_distance,
      'Weighted Euclidean': tfn.distances.weighted_euclidean_distance,
      'Hamming': tfn.distances.hamming_distance,
      'Weighted Hamming': tfn.distances.weighted_hamming_distance,
      'Vertex': tfn.distances.vertex_distance,
      'Tran-Duckstein': tfn.distances.tran_duckstein_distance,
      'L-R': tfn.distances.lr_distance,
      'Mahdavi': tfn.distances.mahdavi_distance
   }

   x = np.array([2, 4, 5])
   y = np.array([1, 2, 3])

   for name, distance in distances.items():
      d = distance(x, y)
      print(f'{name}: {d}')

Defuzzification methods
----------------

To create a crisp ranking from the calculations performed in fuzzy environment, the obtained results should be defuzzified. Different techniques can be used to achieve this. The implemented methods and the example of their usage are presented below.

.. code-block:: python
   :linenos:
   
   defuzzifications = {
      'Mean': tfn.defuzzifications.mean_defuzzification,
      'Mean Area': tfn.defuzzifications.mean_area_defuzzification,
      'Graded Mean Average': tfn.defuzzifications.graded_mean_average_defuzzification,
      'Weighted Mean': tfn.defuzzifications.weighted_mean_defuzzification                                                                                            
   }

   x = np.array([0.2, 0.55, 1.1])

   for name, defuzzy in defuzzifications.items():
      d = defuzzy(x)
      print(f'{name}: {d}')


Weights methods
---------

Criteria weights in multi-criteria problems are responsible for the importance of each parameter taken into consideration. The greater value assigned to the given criterion, the more important it will be in the assessment. For the purpose of weights definition, 4 methods from the library can be used. They are based on the statistical approach, which makes it possible to define the weights objectively, relying only on data diversity.

.. code-block:: python
   :linenos:

   weights_methods = {
      'Equal': f_weights.equal_weights,
      'Shannon Entropy' : f_weights.shannon_entropy_weights,
      'STD': f_weights.standard_deviation_weights,
      'Variance': f_weights.variance_weights
   }

   for name, method in weights_methods.items():
      w = method(random_matrix)
      print(f'{name} \n {w}')


Evaluation 
-----------

Different techniques from the group of Fuzzy Multi-Criteria Decision Analysis methods based on the Triangular Fuzzy Numbers can be used to assess the alternatives. The library contains 10 methods which can be used for this purpose. The examples of their application are presented below.

Decision matrix
----------------

Decision matrix represents the alternatives taken into consideration in the problem. Rows represent amount of alternatives, when columns describes the amount of criteria in the given problem. In the case presented below, we have 4 alternatives and 3 criteria. Moreover, all elements in the matrix should be represent as the Triangular Fuzzy Number.

.. code-block:: python
   :linenos:

   matrix = np.array([
      [[3, 4, 5],[4, 5, 6],[8, 9, 9]],
      [[6, 7, 8],[4, 5, 6],[1, 2, 3]],
      [[5, 6, 7],[2, 3, 4],[3, 4, 5]],
      [[5, 6, 8],[2, 3, 4],[2, 3, 4]],
      [[7, 8, 9],[7, 8, 9],[5, 6, 7]],
   ])


Weights
---------

Weights can be defined objectively, as shown above with the given examples. However, the weights can be also defined directly based on expert knowledge. The library is implemented in a way to handle both crisp and fuzzy weights. Amount of weights should equal the criteria amount. They can be determined as follow.

.. code-block:: python
   :linenos:

   crisp_weights = np.array([0.4, 0.4, 0.2])
   fuzzy_weights = np.array([[5, 7, 9], [7, 9, 9], [3, 5, 7]])


Criteria
---------

Criteria types are ment to reflect the direction of the values that is preferable in the problem. If the values for given criterion should be as big as possible, it is then a profit type and represent as `1` in the criteria types array. If the values should be as low as possible, it is then cost and should be represent as `-1` in the array. Moreover, the criteria types amount should equal amount of criteria in the decision matrix.

.. code-block:: python
   :linenos:

   types = np.array([1, -1, 1])

   
Fuzzy ARAS
-----------

.. code-block:: python
   :linenos:

   f_aras = methods.fARAS()

Fuzzy ARAS evaluation results with crisp and fuzzy weights  

.. code-block:: python
   :linenos:

   print(f'Crisp weights: {f_aras(matrix, crisp_weights, types)}')
   print(f'Fuzzy weights: {f_aras(matrix, fuzzy_weights, types)}')

We can also use ARAS method with different normalizations. Default, it is a `sum_normalization`.

.. code-block:: python
   :linenos:

   aras = {
      'Sum': methods.fARAS(tfn.normalizations.sum_normalization),
      'Max': methods.fARAS(tfn.normalizations.max_normalization),
      'Linear': methods.fARAS(tfn.normalizations.linear_normalization),
      'Minmax': methods.fARAS(tfn.normalizations.minmax_normalization),
      'Vector': methods.fARAS(tfn.normalizations.vector_normalization),
      'SAW': methods.fARAS(tfn.normalizations.saw_normalization)
   }

For every normalization technique, we can perform assessment to obtain results and check if the type of normalization impacts the outcome.

.. code-block:: python
   :linenos:

   results = {}
   for name, function in aras.items():
      results[name] = function(matrix, fuzzy_weights, types)
   
   print(tabulate([[name, *np.round(pref, 2)] for name, pref in results.items()],
      headers=['Method'] + [f'A{i+1}' for i in range(10)]))

We can see that different preferences are obtained with different normalizations. To check if the alternatives are ranked at the same place despite used normalization method, we can use the method from the library called `rank` which calculates ascending or descending position order based on given array. Since the ARAS method assess better alternatives with higher values, the order should be descending.    

.. code-block:: python
   :linenos:

   print(tabulate([[name, *rank(pref, descending=True)] for name, pref in results.items()], 
      headers=['Method'] + [f'A{i+1}' for i in range(10)]))

It can be seen, that the ranking of alternatives is different for different normalization techniques. So the user should bear in mind that different methods can have impact the final result obtained within selected evaluation method.

Fuzzy CODAS
-----------

.. code-block:: python
   :linenos:

   f_codas = methods.fCODAS()
   print(f_codas(matrix, fuzzy_weights, types)) 

Within the CODAS method we can also use different normalizations, as it was in the ARAS method. In addition, we can use different distance metrics to calculate the alternatives preference. Default the `distance_1` is the `euclidean_distance` and `distance_2` is the `hamming_distance`. While calling the fuzzy CODAS object, the `tau` parameter can be given, which is set to `0.02` as default. It is treated as the threshold parameter while calculating the relative assessment matrix. CODAS also assessed better alternatives with higher preferences.

.. code-block:: python
   :linenos:

   codas = {
      'Pair 1': methods.fCODAS(distance_1=tfn.distances.euclidean_distance, distance_2=tfn.distances.hamming_distance),
      'Pair 2': methods.fCODAS(distance_1=tfn.distances.weighted_euclidean_distance, distance_2=tfn.distances.weighted_hamming_distance),
      'Pair 3': methods.fCODAS(distance_1=tfn.distances.vertex_distance, distance_2=tfn.distances.lr_distance),
      'Pair 4': methods.fCODAS(distance_1=tfn.distances.mahdavi_distance, distance_2=tfn.distances.lr_distance),  
   }

Now, when we defined the CODAS object with different pairs of distances, we can calculate the results.

.. code-block:: python
   :linenos:

   results = {}
   for name, function in codas.items():
      results[name] = function(matrix, fuzzy_weights, types)

   print(tabulate([[name, *np.round(pref, 2)] for name, pref in results.items()],
      headers=['Method'] + [f'A{i+1}' for i in range(10)]))

We can see, that different distance metrics also have impact on the final results.

Fuzzy COPRAS
-----------

.. code-block:: python
   :linenos:

   f_copras = methods.fCOPRAS()
   print(f_copras(matrix, fuzzy_weights, types))

As in the case of the ARAS method, in the COPRAS technique, we can also modify the used normalization method. The `saw_normalization` is set as default. Similarly to previous methods, better alternatives are assessed with higher preferences.

.. code-block:: python
   :linenos:

   copras = {
      'Sum': methods.fCOPRAS(tfn.normalizations.sum_normalization),
      'Max': methods.fCOPRAS(tfn.normalizations.max_normalization),
      'Linear': methods.fCOPRAS(tfn.normalizations.linear_normalization),
      'Minmax': methods.fCOPRAS(tfn.normalizations.minmax_normalization),
      'Vector': methods.fCOPRAS(tfn.normalizations.vector_normalization),
      'SAW': methods.fCOPRAS(tfn.normalizations.saw_normalization)
   }

   results = {}
   for name, function in copras.items():
      results[name] = function(matrix, fuzzy_weights, types)

   print(tabulate([[name, *np.round(pref, 2)] for name, pref in results.items()],
      headers=['Method'] + [f'A{i+1}' for i in range(10)]))


Fuzzy EDAS
-----------

.. code-block:: python
   :linenos:

   f_edas = methods.fEDAS()
   print(f_edas(matrix, fuzzy_weights, types))

   In case of using the fuzzy EDAS method, we can modify the used defuzzification technique. Default, the fEDAS method has set the defuzzification to the `mean_defuzzification`. EDAS also evaluate better alternatives with higher preferences.

.. code-block:: python
   :linenos:

   edas = {
      'Mean': methods.fEDAS(defuzzify=tfn.defuzzifications.mean_defuzzification),
      'Mean Area': methods.fEDAS(defuzzify=tfn.defuzzifications.mean_area_defuzzification),
      'Graded Mean Average': methods.fEDAS(defuzzify=tfn.defuzzifications.graded_mean_average_defuzzification),
      'Weighted Mean': methods.fEDAS(defuzzify=tfn.defuzzifications.weighted_mean_defuzzification)                                                                                            
   }

After fEDAS object definition, we can calculate the results based on using different defuzzification methods.

.. code-block:: python
   :linenos:

   results = {}
   for name, function in edas.items():
      results[name] = function(matrix, fuzzy_weights, types)

   print(tabulate([[name, *np.round(pref, 2)] for name, pref in results.items()],
      headers=['Method'] + [f'A{i+1}' for i in range(10)]))

It can be noticed that the results are highly similar while using different methods to defuzzify fuzzy numbers and obtain crisp values.


Fuzzy MABAC
-----------

.. code-block:: python
   :linenos:

   f_mabac = methods.fMABAC()
   print(f_mabac(matrix, fuzzy_weights, types))

While using the fuzzy MABAC method, the normalization and defuzzification methods can be adjusted. Default, normalization is set to `minmax_normalization` and defuzzify to `mean_defuzzification`. MABAC classify better alternatives with higher preferences

.. code-block:: python
   :linenos:

   mabac = {
      'Sum': methods.fMABAC(tfn.normalizations.sum_normalization),
      'Max': methods.fMABAC(tfn.normalizations.max_normalization),
      'Linear': methods.fMABAC(tfn.normalizations.linear_normalization),
      'Minmax': methods.fMABAC(tfn.normalizations.minmax_normalization),
      'Vector': methods.fMABAC(tfn.normalizations.vector_normalization),
      'SAW': methods.fMABAC(tfn.normalizations.saw_normalization)
   }

   results = {}
   for name, function in mabac.items():
      results[name] = function(matrix, fuzzy_weights, types)

   print(tabulate([[name, *np.round(pref, 2)] for name, pref in results.items()],
      headers=['Method'] + [f'A{i+1}' for i in range(10)]))

Again we can see, that different techniques used in the assessment have impact on the final result from the fuzzy MCDA method.

Fuzzy MAIRCA
-----------

.. code-block:: python
   :linenos:

   f_mairca = methods.fMAIRCA()
   print(f_mairca(matrix, fuzzy_weights, types))

Fuzzy MAIRCA method allows for adjusting the parameters responsible for the normalization and the distance measures. Default settings covers the `vector_normalization` and the `vertex_distance`. MAIRCA assigns higher preference values to better classified alternatives.

.. code-block:: python
   :linenos:

   mairca = {
      'Euclidean': methods.fMAIRCA(distance=tfn.distances.euclidean_distance),
      'Weighted Euclidean': methods.fMAIRCA(distance=tfn.distances.weighted_euclidean_distance),
      'Hamming': methods.fMAIRCA(distance=tfn.distances.hamming_distance),
      'Weighted Hamming': methods.fMAIRCA(distance=tfn.distances.weighted_hamming_distance),
      'Vertex': methods.fMAIRCA(distance=tfn.distances.vertex_distance),
      'Tran-Duckstein': methods.fMAIRCA(distance=tfn.distances.tran_duckstein_distance),
      'L-R': methods.fMAIRCA(distance=tfn.distances.lr_distance),
      'Mahdavi': methods.fMAIRCA(distance=tfn.distances.mahdavi_distance)
   }

   results = {}
   for name, function in mairca.items():
      results[name] = function(matrix, fuzzy_weights, types)

   print(tabulate([[name, *np.round(pref, 2)] for name, pref in results.items()],
      headers=['Method'] + [f'A{i+1}' for i in range(10)]))


Fuzzy MOORA
-----------

.. code-block:: python
   :linenos:

   f_moora = methods.fMOORA()
   print(f_moora(matrix, fuzzy_weights, types))

Fuzzy MOORA assigns higher preferences to better alternatives. It allows for the modification of the normalization technique, and the default method is set to `vector_normalization`.

.. code-block:: python
   :linenos:

   moora = {
      'Sum': methods.fMOORA(tfn.normalizations.sum_normalization),
      'Max': methods.fMOORA(tfn.normalizations.max_normalization),
      'Linear': methods.fMOORA(tfn.normalizations.linear_normalization),
      'Minmax': methods.fMOORA(tfn.normalizations.minmax_normalization),
      'Vector': methods.fMOORA(tfn.normalizations.vector_normalization),
      'SAW': methods.fMOORA(tfn.normalizations.saw_normalization)
   }

   results = {}
   for name, function in moora.items():
      results[name] = function(matrix, fuzzy_weights, types)

   print(tabulate([[name, *np.round(pref, 2)] for name, pref in results.items()],
      headers=['Method'] + [f'A{i+1}' for i in range(10)]))


Fuzzy OCRA
-----------

.. code-block:: python
   :linenos:

   f_ocra = methods.fOCRA()
   print(f_ocra(matrix, fuzzy_weights, types))

Fuzzy OCRA has one parameter that can be changed during the evaluation. It is the defuzzification method, which default is set to `mean_defuzzification`. OCRA also assess better alternatives with higher preference values.

.. code-block:: python
   :linenos:

   ocra = {
      'Mean': methods.fOCRA(defuzzify=tfn.defuzzifications.mean_defuzzification),
      'Mean Area': methods.fOCRA(defuzzify=tfn.defuzzifications.mean_area_defuzzification),
      'Graded Mean Average': methods.fOCRA(defuzzify=tfn.defuzzifications.graded_mean_average_defuzzification),
      'Weighted Mean': methods.fOCRA(defuzzify=tfn.defuzzifications.weighted_mean_defuzzification)                                                                                            
   }

   results = {}
   for name, function in ocra.items():
      results[name] = function(matrix, fuzzy_weights, types)

   print(tabulate([[name, *np.round(pref, 2)] for name, pref in results.items()],
      headers=['Method'] + [f'A{i+1}' for i in range(10)]))

Fuzzy TOPSIS
-----------

.. code-block:: python
   :linenos:

   f_topsis = methods.fTOPSIS()
   print(f_topsis(matrix, fuzzy_weights, types))

Fuzzy TOPSIS technique allows for adjusting the parameters responsible for the normalization and the distance calculation. Default methods are set to `linear_normalization` and `vertex_distance`. TOPSIS assures, that better alternatives have higher preferences values. 

.. code-block:: python
   :linenos:

   topsis = {
      'Sum': methods.fTOPSIS(tfn.normalizations.sum_normalization),
      'Max': methods.fTOPSIS(tfn.normalizations.max_normalization),
      'Linear': methods.fTOPSIS(tfn.normalizations.linear_normalization),
      'Minmax': methods.fTOPSIS(tfn.normalizations.minmax_normalization),
      'Vector': methods.fTOPSIS(tfn.normalizations.vector_normalization),
      'SAW': methods.fTOPSIS(tfn.normalizations.saw_normalization)
   }

   results = {}
   for name, function in topsis.items():
      results[name] = function(matrix, fuzzy_weights, types)

   print(tabulate([[name, *np.round(pref, 2)] for name, pref in results.items()],
      headers=['Method'] + [f'A{i+1}' for i in range(10)]))


Fuzzy VIKOR
-----------

.. code-block:: python
   :linenos:

   f_vikor = methods.fVIKOR()
   res = f_vikor(matrix, fuzzy_weights, types)
   print(f'S: {res[0]}')
   print(f'R: {res[1]}')
   print(f'Q: {res[2]}')

Fuzzy VIKOR method is characterized by returning three assessment vectors (S, R, Q). The difference between them lays in the way how they are calculated in the final phase of the evaluation. The VIKOR method performance can be adjusted with the defuzzification method, and the default settings for this parameter is `mean_area_defuzzification`. Moreover, while calling the fVIKOR object, the `v` parameter can be given, which translates how the weight of the strategy will behave. It is set to `0.5` as default. VIKOR ranking can be calculated by sorting the preferences in the ascending order, so in the `rank` method, the parameter should be sey as `descending=False`.

.. code-block:: python
   :linenos:

   vikor = {
      'Mean': methods.fVIKOR(defuzzify=tfn.defuzzifications.mean_defuzzification),
      'Mean Area': methods.fVIKOR(defuzzify=tfn.defuzzifications.mean_area_defuzzification),
      'Graded Mean Average': methods.fVIKOR(defuzzify=tfn.defuzzifications.graded_mean_average_defuzzification),
      'Weighted Mean': methods.fVIKOR(defuzzify=tfn.defuzzifications.weighted_mean_defuzzification)                                                                                            
   }

   results = {}
   for name, function in vikor.items():
      results[name] = function(matrix, fuzzy_weights, types)

   print(tabulate([[name, *np.round(pref[0], 2)] for name, pref in results.items()],
      headers=['Method'] + [f'A{i+1}' for i in range(10)]))


Correlation
-------------

Correlation coefficients can be used to indicate the results similarity. They are based on preference or ranking comparison obtained from the multi-criteria assessment. In the library there are available 4 different measures and the example of their usage is presented below. The `pearson_coef` and `spearman_coef` are ment to be used to compare the preference values, while `weighted_spearman_coef` and `ws_rank_similarity_coef` can be used to compare rankings.

.. code-block:: python
   :linenos:

   x = np.array([0.69, 0.53, 0.76, 0.81, 0.8])
   y = np.array([0.66, 0.54, 0.71, 0.84, 0.77])

   print(f'Spearman: {corrs.spearman_coef(x, y)}')
   print(f'Pearson: {corrs.pearson_coef(x, y)}')

   x = np.array([1, 2, 3, 4, 5])
   y = np.array([2, 1, 3, 4, 5])

   print(f'Weighted Spearman: {corrs.weighted_spearman_coef(x, y)}')
   print(f'WS rank similarity: {corrs.ws_rank_similarity_coef(x, y)}')
