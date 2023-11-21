# Copyright (c) 2022-2023 Jakub Więckowski

__all__ = [
    'mean_defuzzification',
    'mean_area_defuzzification',
    'graded_mean_average_defuzzification',
    'weighted_mean_defuzzification',
    'bisector_defuzzification',
    'height_defuzzification',
    'lom_defuzzification',
    'som_defuzzification',
]


def mean_defuzzification(a):
    """
        Defuzzify the Triangular Fuzzy Number into crisp value.
        Uses a formula 1/3 * (l + m + r)

        Parameters
        ----------
            a : ndarray
                Triangular Fuzzy Number

        Returns
        -------
            float
                Crisp value
    """

    return 1/3 * (a[0] + a[1] + a[2])


def mean_area_defuzzification(a):
    """
        Defuzzify the Triangular Fuzzy Number into crisp value.
        Uses a formula 1/4 * (l + 2m + r)

        Parameters
        ----------
            a : ndarray
                Triangular Fuzzy Number

        Returns
        -------
            float
                Crisp value
    """

    return 1/4 * (a[0] + 2 * a[1] + a[2])


def graded_mean_average_defuzzification(a):
    """
        Defuzzify the Triangular Fuzzy Number into crisp value.
        Uses a formula 1/6 * (l + 4m + r)

        Parameters
        ----------
            a : ndarray
                Triangular Fuzzy Number

        Returns
        -------
            float
                Crisp value
    """

    return 1/6 * (a[0] + 4 * a[1] + a[2])


def weighted_mean_defuzzification(a, k=2):
    """
        Defuzzify the Triangular Fuzzy Number into crisp value.
        Uses a formula (m + (r - m) - (m - l)) / (k + 2)

        Parameters
        ----------
            a : ndarray
                Triangular Fuzzy Number

            k : int
                weight factor

        Returns
        -------
            float
                Crisp value
    """
    return a[1] + ((a[2] - a[1]) - (a[1] - a[0])) / (k + 2)

def bisector_defuzzification(a):
    """
        Defuzzify the Triangular Fuzzy Number into crisp value.
        Uses a formula (l+m) / 2

        Parameters
        ----------
            a : ndarray
                Triangular Fuzzy Number
        Returns
        -------
            float
                Crisp value
    """
    return (a[0] + a[2]) / 2

def height_defuzzification(a):
    """
        Defuzzify the Triangular Fuzzy Number into crisp value.
        Returns the middle element of Triangular Fuzzy Number as result

        Parameters
        ----------
            a : ndarray
                Triangular Fuzzy Number
        Returns
        -------
            float
                Crisp value
    """
    return a[1]

def lom_defuzzification(a):
    """
        Defuzzify the Triangular Fuzzy Number into crisp value.
        Largest of Maximum (LOM) defuzzification method.
        Returns the maximum value of Triangular Fuzzy Number

        Parameters
        ----------
            a : ndarray
                Triangular Fuzzy Number
        Returns
        -------
            float
                Crisp value
    """
    return max(a)

def som_defuzzification(a):
    """
        Defuzzify the Triangular Fuzzy Number into crisp value.
        Smallest of Maximum (SOM) defuzzification method.
        Returns the minimum value of Triangular Fuzzy Number

        Parameters
        ----------
            a : ndarray
                Triangular Fuzzy Number
        Returns
        -------
            float
                Crisp value
    """
    return min(a)
