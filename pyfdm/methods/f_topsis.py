# Copyright (c) 2022 Jakub Więckowski

from .topsis.fuzzy import fuzzy
from .fuzzy_sets.tfn.normalizations import linear_normalization
from .fuzzy_sets.tfn.distances import vertex_distance

from .validator import Validator


class fTOPSIS():
    def __init__(self, normalization=linear_normalization, distance=vertex_distance):
        """
            Creates fuzzy TOPSIS method object with linear normalization and vertex distance function

            Parameters
            ----------
                normalization: callable
                    Function used to normalize the decision matrix

                distance: callable
                    Function used to calculate distance from fuzzy negative solution

        """

        self.normalization = normalization
        self.distance = distance

    def __call__(self, matrix, weights, types):
        """
            Calculates the alternatives preferences

            Parameters
            ----------
                matrix : ndarray
                    Decision matrix / alternatives data.
                    Alternatives are in rows and Criteria are in columns.

                weights : ndarray
                    Vector of criteria weights in a crisp form

                types : ndarray
                    Types of criteria, 1 profit, -1 cost

            Returns
            ----------
                ndarray:
                    Preference calculated for alternatives. Greater values are placed higher in ranking
        """
        # validate data
        Validator.fuzzy_validation(matrix, weights)

        return fuzzy(matrix, weights, types, self.normalization, self.distance).astype(float)
