# Copyright (c) 2022 Jakub Więckowski

from .topsis.fuzzy import fuzzy
from .utils.normalizations import linear_normalization
from .utils.distances import vertex_distance
from ..helpers import rank, normalize_weights

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

        self.preferences = fuzzy(matrix, normalize_weights(weights), types, self.normalization, self.distance).astype(float)
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