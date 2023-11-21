# Copyright (c) 2022 Jakub Więckowski

from .codas.fuzzy import fuzzy
from .utils.normalizations import max_normalization
from .utils.distances import euclidean_distance, hamming_distance
from ..helpers import rank

from .validator import Validator


class fCODAS():
    def __init__(self, normalization=max_normalization, distance_1=euclidean_distance, distance_2=hamming_distance):
        """
            Create fuzzy CODAS method object with max normalization function and Euclidean and Hamming distances metrics

            Parameters
            ----------
                normalization: callable
                    Function used to calculate normalized decision matrix

                distance_1: callable
                    Function used to calculate distance from fuzzy negative solution

                distance_2: callable
                    Function used to calculate distance form fuzzy negative solution

        """

        self.normalization = normalization
        self.distance_1 = distance_1
        self.distance_2 = distance_2
        self.__descending = True

    def __call__(self, matrix, weights, types, tau=0.02, *args, **kwargs):
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

                tau: float, default = 0.02
                    Threshold parameter

            Returns
            ----------
                ndarray:
                    Preference calculated for alternatives. Greater values are placed higher in ranking
        """
        # validate data
        Validator.fuzzy_validation(matrix, weights)

        self.preferences = fuzzy(matrix, weights, types, self.normalization, self.distance_1, self.distance_2, tau).astype(float)
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