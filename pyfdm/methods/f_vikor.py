# Copyright (c) 2022 Jakub Więckowski

import numpy as np
from .vikor.fuzzy import fuzzy
from .fuzzy_sets.tfn.defuzzifications import mean_area_defuzzification
from ..helpers import rank

from .validator import Validator


class fVIKOR():
    def __init__(self, defuzzify=mean_area_defuzzification):
        """
            Creates fuzzy VIKOR method object with mean area defuzzification function

            Parameters
            ----------
                defuzzify: callable
                    Function used to defuzzify the TFN into crisp value

        """

        self.defuzzify = defuzzify
        self.__descending = False

    def __call__(self, matrix, weights, types, v=0.5):
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

                v : float
                    Weight of the strategy (see VIKOR algorithm explanation).

            Returns
            ----------
                ndarray:
                    Preference calculated for alternatives. Lower values are placed higher in ranking
        """
        # validate data
        Validator.fuzzy_validation(matrix, weights)

        self.preferences = fuzzy(matrix, weights, types, self.defuzzify, v)
        return self.preferences

    def rank(self):
        """
            Calculates the alternatives ranking based on the obtained preferences

            Returns
            ----------
                ndarray:
                    Rankings of alternatives for S, R, Q approaches
        """
        try:
            return np.array([rank(pref, self.__descending) for pref in self.preferences])
        except AttributeError:
            raise AttributeError('Cannot calculate ranking before assessment')
        except:
            raise ValueError('Error occurred in ranking calculation')
    
