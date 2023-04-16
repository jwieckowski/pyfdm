# Copyright (c) 2022 Jakub Więckowski

from .copras.fuzzy import fuzzy
from .fuzzy_sets.tfn.normalizations import saw_normalization
from ..helpers import rank

from .validator import Validator


class fCOPRAS():
    def __init__(self, normalization=saw_normalization):
        """
            Create fuzzy COPRAS method object with saw normalization function

            Parameters
            ----------
                normalization: callable
                    Function used to calculate normalized decision matrix

        """

        self.normalization = normalization
        self.__descending = True

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

        self.preferences = fuzzy(matrix, weights, types, self.normalization).astype(float)
        return self.preferences

    def rank(self):
        """
            Calculates the alternatives ranking based on the obtained preferences

            Returns
            ----------
                ndarray:
                    Ranking of alternatives
        """
        try:
            return rank(self.preferences, self.__descending)
        except AttributeError:
            raise AttributeError('Cannot calculate ranking before assessment')
        except:
            raise ValueError('Error occurred in ranking calculation')