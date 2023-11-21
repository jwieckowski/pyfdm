# Copyright (c) 2022-2023 Jakub Więckowski

import numpy as np

__all__ = [
    'canberra_distance',
    'chebyshev_distance',
    'euclidean_distance',
    'hamming_distance',
    'lr_distance',
    'mahdavi_distance',
    'tran_duckstein_distance',
    'vertex_distance',
    'weighted_euclidean_distance',
    'weighted_hamming_distance',
]


def euclidean_distance(a, b):
    """
        Calculates the distance between two Triangular Fuzzy Numbers using Euclidean distance

        Parameters
        ----------
            a : ndarray
                Triangular Fuzzy Number

            b : ndarray
                Triangular Fuzzy Number

        Returns
        -------
            float
                Crisp value representing distance
    """
    return np.sqrt(((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2))


def weighted_euclidean_distance(a, b):
    """
        Calculates the distance between two Triangular Fuzzy Numbers using weighted Euclidean distance

        Parameters
        ----------
            a : ndarray
                Triangular Fuzzy Number

            b : ndarray
                Triangular Fuzzy Number

        Returns
        -------
            float
                Crisp value representing distance
    """
    return np.sqrt(((a[0] - b[0])**2 + 2*(a[1] - b[1])**2 + (a[2] - b[2])**2) / 4)


def hamming_distance(a, b):
    """
        Calculates the distance between two Triangular Fuzzy Numbers using Hamming distance

        Parameters
        ----------
            a : ndarray
                Triangular Fuzzy Number

            b : ndarray
                Triangular Fuzzy Number

        Returns
        -------
            float
                Crisp value representing distance
    """
    return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1]) + np.abs(a[2] - b[2])


def weighted_hamming_distance(a, b):
    """
        Calculates the distance between two Triangular Fuzzy Numbers using weighted Hamming distance

        Parameters
        ----------
            a : ndarray
                Triangular Fuzzy Number

            b : ndarray
                Triangular Fuzzy Number

        Returns
        -------
            float
                Crisp value representing distance
    """
    return (np.abs(a[0] - b[0]) + 2*np.abs(a[1] - b[1]) + np.abs(a[2] - b[2])) / 4

def vertex_distance(a, b):
    """
        Calculates the distance between two Triangular Fuzzy Numbers using Vertex distance

        Parameters
        ----------
            a : ndarray
                Triangular Fuzzy Number

            b : ndarray
                Triangular Fuzzy Number

        Returns
        -------
            float
                Crisp value representing distance
    """
    return np.sqrt(((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2) / 3)


def tran_duckstein_distance(a, b):
    """
        Calculates the distance between two Triangular Fuzzy Numbers using Tran Duckstein distance

        Parameters
        ----------
            a : ndarray
                Triangular Fuzzy Number

            b : ndarray
                Triangular Fuzzy Number

        Returns
        -------
            float
                Crisp value representing distance
    """
    return (a[1] - b[1])**2 + 0.5 * (a[1] - b[1]) * ((a[2] - b[0]) - (b[2] - b[0])) + 1/9 * ((a[2]-a[1])**2 + (a[1]-a[0])**2 + (b[2]-b[1])**2 + (b[1]-b[0])**2) - 1/9 * ((a[1]-a[0]) * (a[2]-a[1]) + (b[1]-b[0]) * (b[2]-b[1])) + 1/6 * (2 * a[1] - a[0] - a[2]) * (2 * b[1] - b[0] - b[2])


def lr_distance(a, b, r=0.5):
    """
        Calculates the distance between two Triangular Fuzzy Numbers using L-R distance

        Parameters
        ----------
            a : ndarray
                Triangular Fuzzy Number

            b : ndarray
                Triangular Fuzzy Number

        Returns
        -------
            float
                Crisp value representing distance
    """
    return (a[1] - b[1])**2 + ((a[1] - r * a[0]) - (b[1] - r * b[0]))**2 + ((a[1] + r * a[2]) - (b[1] + r * b[2]))**2


def mahdavi_distance(a, b):
    """
        Calculates the distance between two Triangular Fuzzy Numbers using Mahdavi distance

        Parameters
        ----------
            a : ndarray
                Triangular Fuzzy Number

            b : ndarray
                Triangular Fuzzy Number

        Returns
        -------
            float
                Crisp value representing distance
    """
    return np.sqrt(1/6 * ( np.sum([ (b[i] - a[i])**2 for i in range(3)]) + (b[1] - a[1])**2 + np.sum([(b[i] - a[i]) * (b[i+1] - a[i+1]) for i in range(2)]) ))

def chebyshev_distance(a, b):
    """
    Calculates the Chebyshev distance between two Triangular Fuzzy Numbers.

    Parameters
    ----------
        a : ndarray
            Triangular Fuzzy Number

        b : ndarray
            Triangular Fuzzy Number

    Returns
    -------
        float
            Crisp value representing distance
    """
    distances = [abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2])]
    
    return max(distances)

def canberra_distance(a, b):
    """
    Calculates the Canberra distance between two Triangular Fuzzy Numbers.

    Parameters
    ----------
        a : ndarray
            Triangular Fuzzy Number

        b : ndarray
            Triangular Fuzzy Number

    Returns
    -------
        float
            Crisp value representing distance
    """

    return np.sum([np.abs(a[i] - b[i]) / (np.abs(a[i]) + np.abs(b[i])) for i in range(len(a))])
