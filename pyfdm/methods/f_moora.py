# Copyright (c) 2022 Jakub Więckowski

from .moora.fuzzy import fuzzy
from .fuzzy_sets.tfn.normalizations import vector_normalization

from .validator import Validator


class fMOORA():
    def __init__(self, normalization=vector_normalization):
        """
            Create fuzzy MOORA method object with vector normalization function

            Parameters
            ----------
                normalization: callable
                    Function used to normalize the decision matrix

        """

        self.normalization = normalization

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
        Validator.fuzzy_validation(matrix, weights, types)

        return fuzzy(matrix, weights, types, self.normalization).astype(float)
