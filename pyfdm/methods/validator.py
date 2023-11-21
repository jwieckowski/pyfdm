# Copyright (c) 2022-2023 Jakub Więckowski

import numpy as np

class Validator():

    @staticmethod
    def validate_input(matrix, weights, types):
        """
            Checks if number of criteria, number of weights, and number of types are the same

            Parameters
            ----------
                matrix : ndarray
                    Decision matrix / alternatives data.
                    Alternatives are in rows and Criteria are in columns.

                weights : ndarray
                    Vector of weights in a crisp form

                types : ndarray
                    Types of criteria, 1 profit, -1 cost

            Returns
            -------
                raises:
                    ValueError if shapes of matrix, weights and types are not the same

        """
        if types is not None:
            if len(np.unique([matrix.shape[1], weights.shape[0], types.shape[0]])) != 1:
                raise ValueError(
                    f'Number of criteria should equals number of weights and types, not {matrix.shape[1]}, {weights.shape[0]}, {types.shape[0]}')
        else:
            if len(np.unique([matrix.shape[1], weights.shape[0]])) != 1:
                raise ValueError(
                    f'Number of criteria should equals number of weights, not {matrix.shape[1]}, {weights.shape[0]}')

    @staticmethod
    def validate_tfn_matrix(matrix):
        """
            Checks if TFN matrix is defined properly, all elements should have length of 3

            Parameters
            ----------
                matrix : ndarray
                    Decision matrix / alternatives data.
                    Alternatives are in rows and Criteria are in columns.

            Returns
            -------
                raises:
                    ValueError if matrix elements has different length than 3

        """
        if matrix.ndim != 3 or matrix.shape[2] != 3:
            raise ValueError(
                'TFN matrix elements should all have length of 3')

    @staticmethod
    def validate_weights(weights, crisp_required=False):
        """
            For crisp weights checks if sum of weights equals 1
            For fuzzy weights checks if given as Triangular Fuzzy Numbers

            Parameters
            ----------
                weights : ndarray
                    Vector of weights in a crisp form or as a TFNs

                crisp_required : bool, default=False
                    Flag representing the need to obtain crisp criteria weights as input data

            Returns
            -------
                raises:
                    ValueError if sum of weights is different than 1 or not in TFN form

        """

        if crisp_required:
            if weights.ndim != 1:
                raise ValueError('Criteria weights should be given as crisp values')
        else:
            if weights.ndim == 1:
                if np.round(np.sum(weights), 4) != 1:
                    raise ValueError(
                        f'Sum of crisp weights should equal 1, not {np.sum(weights)}')
            else:
                if weights.ndim != 2 or weights.shape[1] != 3:
                    raise ValueError(
                        'Fuzzy weights should be given as Triangular Fuzzy Numbers')

    @staticmethod
    def validate_types(types):
        """
            Checks if all criteria types are same type

            Parameters
            ----------
                types : ndarray
                    Types of criteria, 1 profit, -1 cost

            Returns
            -------
                raises:
                    ValueError if criteria types are the same

        """
        if types is not None:
            if len(np.unique(types)) == 1:
                raise ValueError('Criteria types should not be the same')

    @staticmethod
    def fuzzy_validation(matrix, weights, types=None, crisp_required=False):
        """
            Runs all validations for the fuzzy TFN extension

            Parameters
            ----------
                matrix : ndarray
                    Decision matrix / alternatives data.
                    Alternatives are in rows and Criteria are in columns.

                weights : ndarray
                    Vector of weights in a crisp form

                types : ndarray, default=None
                    Types of criteria, 1 profit, -1 cost

                crisp_required : bool, default=False
                    Flag representing the need to obtain crisp criteria weights as input data

            Returns
            -------
                raises:
                    ValueError if one of validations do not pass

        """
        Validator.validate_input(matrix, weights, types)
        Validator.validate_tfn_matrix(matrix)
        Validator.validate_weights(weights, crisp_required)
        Validator.validate_types(types)
