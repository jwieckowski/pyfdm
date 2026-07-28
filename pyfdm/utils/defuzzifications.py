# Copyright (c) 2022 - 2026 Jakub Więckowski

import numpy as np

from ..validator import Validator

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

def mean_defuzzification(a: np.ndarray | list) -> float:
    """
    Defuzzify a Triangular Fuzzy Number using the arithmetic mean.

    The crisp value is computed as

    .. math::

        \\frac{l + m + u}{3}

    Parameters
    ----------
    a : array-like
        Triangular Fuzzy Number ``(l, m, u)``.

    Returns
    -------
    float
        Defuzzified crisp value.
    """

    Validator.validate_tfn(a, 'a')

    try:
        return (a[0] + a[1] + a[2]) / 3
    except Exception as e:
        raise RuntimeError(f"Failed to perform Mean defuzzification: {e}") from e

def mean_area_defuzzification(a: np.ndarray | list) -> float:
    """
    Defuzzify a Triangular Fuzzy Number using the mean area method.

    The crisp value is computed as

    .. math::

        \\frac{l + 2m + u}{4}

    Parameters
    ----------
    a : array-like
        Triangular Fuzzy Number.

    Returns
    -------
    float
        Defuzzified crisp value.
    """

    Validator.validate_tfn(a, 'a')

    try:
        return (a[0] + 2 * a[1] + a[2]) / 4
    except Exception as e:
        raise RuntimeError(f"Failed to perform Mean area defuzzification: {e}") from e

def graded_mean_average_defuzzification(a: np.ndarray | list) -> float:
    """
    Defuzzify a Triangular Fuzzy Number using the graded mean average.

    The crisp value is computed as

    .. math::

        \\frac{l + 4m + u}{6}

    Parameters
    ----------
    a : array-like
        Triangular Fuzzy Number.

    Returns
    -------
    float
        Defuzzified crisp value.
    """

    Validator.validate_tfn(a, 'a')

    try:
        return (a[0] + 4 * a[1] + a[2]) / 6
    except Exception as e:
        raise RuntimeError(f"Failed to perform Graded mean average defuzzification: {e}") from e

def weighted_mean_defuzzification(a: np.ndarray | list, k: float = 2) -> float:
    """
    Defuzzify a Triangular Fuzzy Number using the weighted mean method.

    The crisp value is computed as

    .. math::

        m + \\frac{(u-m)-(m-l)}{k+2}

    Parameters
    ----------
    a : array-like
        Triangular Fuzzy Number.

    k : float, default=2
        Weight factor.

    Returns
    -------
    float
        Defuzzified crisp value.

    Raises
    ------
    ValueError
        If ``k <= -2``.
    """

    Validator.validate_tfn(a, 'a')

    if k <= -2:
        raise ValueError("'k' must be greater than -2.")

    try:
        return a[1] + ((a[2] - a[1]) - (a[1] - a[0])) / (k + 2)
    except Exception as e:
        raise RuntimeError(f"Failed to perform Weighted mean defuzzification: {e}") from e

def bisector_defuzzification(a: np.ndarray | list) -> float:
    """
    Defuzzify a Triangular Fuzzy Number using the bisector method.

    The crisp value is computed as

    .. math::

        \\frac{l + u}{2}

    Parameters
    ----------
    a : array-like
        Triangular Fuzzy Number.

    Returns
    -------
    float
        Defuzzified crisp value.
    """

    Validator.validate_tfn(a, 'a')

    try:
        return (a[0] + a[2]) / 2
    except Exception as e:
        raise RuntimeError(f"Failed to perform Bisector defuzzification: {e}") from e

def height_defuzzification(a: np.ndarray | list) -> float:
    """
    Defuzzify a Triangular Fuzzy Number using the height method.

    Returns the modal value of the TFN.

    Parameters
    ----------
    a : array-like
        Triangular Fuzzy Number.

    Returns
    -------
    float
        Defuzzified crisp value.
    """

    Validator.validate_tfn(a, 'a')

    try:
        return a[1]
    except Exception as e:
        raise RuntimeError(f"Failed to perform Height defuzzification: {e}") from e

def lom_defuzzification(a: np.ndarray | list) -> float:
    """
    Defuzzify a Triangular Fuzzy Number using the Largest of Maximum (LOM)
    method.

    Parameters
    ----------
    a : array-like
        Triangular Fuzzy Number.

    Returns
    -------
    float
        Largest value of the TFN.
    """

    Validator.validate_tfn(a, 'a')

    try:
        return np.max(a)
    except Exception as e:
        raise RuntimeError(f"Failed to perform LOM defuzzification: {e}") from e

def som_defuzzification(a: np.ndarray | list) -> float:
    """
    Defuzzify a Triangular Fuzzy Number using the Smallest of Maximum (SOM)
    method.

    Parameters
    ----------
    a : array-like
        Triangular Fuzzy Number.

    Returns
    -------
    float
        Smallest value of the TFN.
    """

    Validator.validate_tfn(a, 'a')

    try:
        return np.min(a)
    except Exception as e:
        raise RuntimeError(f"Failed to perform SOM defuzzification: {e}") from e