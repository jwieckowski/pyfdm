# Copyright (c) 2023 Jakub Więckowski, Andrii Shekhovtsov

from .spotis.fuzzy import fuzzy
from ..helpers import rank

from .validator import Validator
import numpy as np

class fSPOTIS():
    def __init__(self, normalization=None):
        """
            Creates fuzzy SPOTIS method object

            Parameters
            ----------
                normalization: callable, default=None
                    Function used to normalize the decision matrix

        """

        self.normalization = normalization
        self.__descending = True

    def __call__(self, matrix, weights, types, bounds, *args, **kwargs):
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
    
                bounds : ndarray
                    Decision problem bounds / criteria bounds. Should be two dimensional array with [min, max] value for in criterion in rows.

            Returns
            ----------
                ndarray:
                    Preference calculated for alternatives. Greater values are placed higher in ranking
        """
        # validate data
        Validator.fuzzy_validation(matrix, weights, types, crisp_required=True)

        isp = bounds[np.arange(bounds.shape[0]), ((types+1)//2).astype('int')]

        self.preferences = fuzzy(matrix, weights, self.normalization, bounds, isp).astype(float)
        return self.preferences

    def make_bounds(self, matrix):

        bounds = np.hstack((
            np.min(matrix[:, :, 0], axis=0).reshape(-1, 1),
            np.max(matrix[:, :, 2], axis=0).reshape(-1, 1),
        ))

        return bounds

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
