# Copyright (c) 2022 Jakub Więckowski

from .mairca.fuzzy import fuzzy
from .fuzzy_sets.tfn.normalizations import vector_normalization
from .fuzzy_sets.tfn.distances import vertex_distance

from .validator import Validator


class fMAIRCA():
    def __init__(self, normalization=vector_normalization, distance=vertex_distance):
        """
            Create fuzzy MAIRCA method object with vector normalization and vertex distance functions

            Parameters
            ----------
                normalization: callable
                    Function used to normalize the decision matrix

                distance: callable
                    Function used to calculate distance between two Triangular Fuzzy Numbers

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
