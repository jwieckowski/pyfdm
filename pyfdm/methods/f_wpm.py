# Copyright (c) 2023 Jakub Więckowski

import numpy as np
from .wpm.fuzzy import fuzzy
from .utils.defuzzifications import mean_defuzzification
from ..helpers import rank

from .validator import Validator

class fWPM():
    def __init__(self, normalization=None, defuzzify=mean_defuzzification):
        """
            Creates fuzzy WPM method object with mean_defuzzification

            Parameters
            ----------
                normalization: callable, default=None
                    Function used to normalize the decision matrix

                defuzzify: callable
                    Function used to defuzzify the TFN into crisp value

        """

        self.normalization = normalization
        self.defuzzify = defuzzify
        self.__descending = True

    def __call__(self, matrix, weights, *args, **kwargs):
        """
            Calculates the alternatives preferences

            Parameters
            ----------
                matrix : ndarray
                    Decision matrix / alternatives data.
                    Alternatives are in rows and Criteria are in columns.

                weights : ndarray
                    Vector of criteria weights in a crisp form

            Returns
            ----------
                ndarray:
                    Preference calculated for alternatives. Lower values are placed higher in ranking
        """
        # validate data
        Validator.fuzzy_validation(matrix, weights)

        self.preferences = fuzzy(matrix, weights, self.normalization, self.defuzzify)
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