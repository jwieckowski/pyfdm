# Copyright (c) 2022 Jakub Więckowski

from .mabac.fuzzy import fuzzy
from .utils.defuzzifications import mean_defuzzification
from .utils.normalizations import minmax_normalization
from ..helpers import rank

from .validator import Validator


class fMABAC():
    def __init__(self, normalization=minmax_normalization, defuzzify=mean_defuzzification):
        """
            Create fuzzy MAIRCA method object with minmax normalization and mean defuzzification functions

            Parameters
            ----------
                normalization: callable
                    Function used to normalize the decision matrix

                defuzzify: callable
                    Function used to defuzzify the TFN into crisp value

        """

        self.normalization = normalization
        self.defuzzify = defuzzify
        self.__descending = True

    def __call__(self, matrix, weights, types, *args, **kwargs):
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

        self.preferences = fuzzy(matrix, weights, types, self.normalization, self.defuzzify).astype(float)
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