# Copyright (c) 2022 Jakub Więckowski

from .edas.fuzzy import fuzzy
from .fuzzy_sets.tfn.defuzzifications import mean_defuzzification

from .validator import Validator


class fEDAS():
    def __init__(self, defuzzify=mean_defuzzification):
        """
            Create fuzzy EDAS method object with mean defuzification function

            Parameters
            ----------
                defuzzify: callable
                    Function used to defuzzify the TFN into crisp value

        """

        self.defuzzify = defuzzify

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

        return fuzzy(matrix, weights, types, self.defuzzify).astype(float)
