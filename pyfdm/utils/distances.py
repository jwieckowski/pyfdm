# Copyright (c) 2022 - 2026 Jakub Więckowski

import numpy as np

from ..validator import Validator

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


def euclidean_distance(
    a: np.ndarray | list,
    b: np.ndarray | list
) -> float:
    """
    Calculate the Euclidean distance between two Triangular Fuzzy Numbers.

    The distance is computed as:

    .. math::

        d(A,B)=\\sqrt{(l_A-l_B)^2+(m_A-m_B)^2+(u_A-u_B)^2}

    Parameters
    ----------
    a : np.ndarray | list
        First Triangular Fuzzy Number ``(l, m, u)``.

    b : np.ndarray | list
        Second Triangular Fuzzy Number ``(l, m, u)``.

    Returns
    -------
    float
        Euclidean distance between the two TFNs.

    Raises
    ------
    ValueError
        If either input is not a valid Triangular Fuzzy Number.

    RuntimeError
        If the distance calculation fails.
    """

    Validator.validate_tfn(a, 'a')
    Validator.validate_tfn(b, 'b')

    try:
        return float(
            np.sqrt(
                (a[0] - b[0]) ** 2
                + (a[1] - b[1]) ** 2
                + (a[2] - b[2]) ** 2
            )
        )

    except Exception as e:
        raise RuntimeError(f"Failed to compute Euclidean distance: {e}") from e

def weighted_euclidean_distance(
    a: np.ndarray | list,
    b: np.ndarray | list
) -> float:
    """
    Calculate the weighted Euclidean distance between two
    Triangular Fuzzy Numbers.

    The distance is computed as:

    .. math::

        d(A,B)=
        \\sqrt{
        \\frac{
        (l_A-l_B)^2+
        2(m_A-m_B)^2+
        (u_A-u_B)^2
        }{4}
        }

    The middle value of the TFN is assigned a higher importance.

    Parameters
    ----------
    a : np.ndarray | list
        First Triangular Fuzzy Number ``(l, m, u)``.

    b : np.ndarray | list
        Second Triangular Fuzzy Number ``(l, m, u)``.

    Returns
    -------
    float
        Weighted Euclidean distance between the two TFNs.

    Raises
    ------
    ValueError
        If either input is not a valid Triangular Fuzzy Number.

    RuntimeError
        If the distance calculation fails.
    """

    Validator.validate_tfn(a, 'a')
    Validator.validate_tfn(b, 'b')

    try:
        return float(
            np.sqrt(
                (
                    (a[0] - b[0]) ** 2
                    + 2 * (a[1] - b[1]) ** 2
                    + (a[2] - b[2]) ** 2
                ) / 4
            )
        )

    except Exception as e:
        raise RuntimeError(f"Failed to compute weighted Euclidean distance: {e}") from e

def hamming_distance(
    a: np.ndarray | list,
    b: np.ndarray | list
) -> float:
    """
    Calculate the Hamming distance between two Triangular Fuzzy Numbers.

    The distance is computed as the sum of absolute differences between
    corresponding TFN components:

    .. math::

        d(A,B)=|l_A-l_B|+|m_A-m_B|+|u_A-u_B|

    Parameters
    ----------
    a : np.ndarray | list
        First Triangular Fuzzy Number ``(l, m, u)``.

    b : np.ndarray | list
        Second Triangular Fuzzy Number ``(l, m, u)``.

    Returns
    -------
    float
        Hamming distance between the two TFNs.

    Raises
    ------
    ValueError
        If either input is not a valid Triangular Fuzzy Number.

    RuntimeError
        If the distance calculation fails.
    """

    Validator.validate_tfn(a, 'a')
    Validator.validate_tfn(b, 'b')

    try:
        return float(
            np.sum(np.abs(a - b))
        )

    except Exception as e:
        raise RuntimeError(f"Failed to compute Hamming distance: {e}") from e

def weighted_hamming_distance(
    a: np.ndarray | list,
    b: np.ndarray | list
) -> float:
    """
    Calculate the weighted Hamming distance between two
    Triangular Fuzzy Numbers.

    The distance is computed as:

    .. math::

        d(A,B)=
        \\frac{
        |l_A-l_B|+
        2|m_A-m_B|+
        |u_A-u_B|
        }{4}

    The middle value of the TFN receives a higher contribution.

    Parameters
    ----------
    a : np.ndarray | list
        First Triangular Fuzzy Number ``(l, m, u)``.

    b : np.ndarray | list
        Second Triangular Fuzzy Number ``(l, m, u)``.

    Returns
    -------
    float
        Weighted Hamming distance between the two TFNs.

    Raises
    ------
    ValueError
        If either input is not a valid Triangular Fuzzy Number.

    RuntimeError
        If the distance calculation fails.
    """

    Validator.validate_tfn(a, 'a')
    Validator.validate_tfn(b, 'b')

    try:
        return float(
            (
                np.abs(a[0] - b[0])
                + 2 * np.abs(a[1] - b[1])
                + np.abs(a[2] - b[2])
            ) / 4
        )

    except Exception as e:
        raise RuntimeError(f"Failed to compute weighted Hamming distance: {e}") from e

def vertex_distance(
    a: np.ndarray | list,
    b: np.ndarray | list
) -> float:
    """
    Calculate the Vertex distance between two Triangular Fuzzy Numbers.

    The Vertex method computes the Euclidean distance between two TFNs
    considering the three characteristic points and normalizes the result:

    .. math::

        d(A,B)=
        \\sqrt{
        \\frac{
        (l_A-l_B)^2+
        (m_A-m_B)^2+
        (u_A-u_B)^2
        }{3}
        }

    Parameters
    ----------
    a : np.ndarray | list
        First Triangular Fuzzy Number ``(l, m, u)``.

    b : np.ndarray | list
        Second Triangular Fuzzy Number ``(l, m, u)``.

    Returns
    -------
    float
        Vertex distance between the two TFNs.

    Raises
    ------
    ValueError
        If either input is not a valid Triangular Fuzzy Number.

    RuntimeError
        If the distance calculation fails.
    """

    Validator.validate_tfn(a, 'a')
    Validator.validate_tfn(b, 'b')

    try:
        return float(np.sqrt(np.sum((a - b) ** 2) / 3))

    except Exception as e:
        raise RuntimeError(f"Failed to compute Vertex distance: {e}") from e

def chebyshev_distance(
    a: np.ndarray | list,
    b: np.ndarray | list
) -> float:
    """
    Calculate the Chebyshev distance between two Triangular Fuzzy Numbers.

    The Chebyshev distance is defined as the maximum absolute difference
    between corresponding components of two TFNs.

    Parameters
    ----------
    a : np.ndarray | list
        First Triangular Fuzzy Number represented as (l, m, u).

    b : np.ndarray | list
        Second Triangular Fuzzy Number represented as (l, m, u).

    Returns
    -------
    float
        Crisp distance value.

    Raises
    ------
    ValueError
        If inputs cannot be converted to numeric TFNs or do not contain
        exactly three elements.
    RuntimeError
        If the distance computation fails.
    """

    Validator.validate_tfn(a, 'a')
    Validator.validate_tfn(b, 'b')

    try:
        distances = np.abs(a - b)
        return float(np.max(distances))

    except Exception as e:
        raise RuntimeError(f"Failed to compute Chebyshev distance: {e}") from e


def canberra_distance(
    a: np.ndarray | list,
    b: np.ndarray | list
) -> float:
    """
    Calculate the Canberra distance between two Triangular Fuzzy Numbers.

    Canberra distance measures the sum of normalized absolute differences
    between corresponding TFN components:

    .. math::

        d(A,B)=\\sum_i \\frac{|a_i-b_i|}
        {|a_i|+|b_i|}

    Parameters
    ----------
    a : np.ndarray | list
        First Triangular Fuzzy Number represented as (l, m, u).

    b : np.ndarray | list
        Second Triangular Fuzzy Number represented as (l, m, u).

    Returns
    -------
    float
        Crisp distance value.

    Raises
    ------
    ValueError
        If inputs are invalid.
    RuntimeError
        If the distance computation fails.
    """

    Validator.validate_tfn(a, 'a')
    Validator.validate_tfn(b, 'b')

    try:
        denominator = np.abs(a) + np.abs(b)

        denominator = np.where(
            denominator == 0,
            1e-10,
            denominator
        )

        return float(np.sum(np.abs(a - b) / denominator))

    except Exception as e:
        raise RuntimeError(f"Failed to compute Canberra distance: {e}") from e

def lr_distance(
    a: np.ndarray | list,
    b: np.ndarray | list,
    r: float = 0.5
) -> float:
    """
    Calculate the L-R distance between two Triangular Fuzzy Numbers.

    The L-R distance considers the modal value and the left/right spreads
    of TFNs. The parameter ``r`` controls the influence of the lower and
    upper components.

    Parameters
    ----------
    a : np.ndarray | list
        First Triangular Fuzzy Number represented as (l, m, u).

    b : np.ndarray | list
        Second Triangular Fuzzy Number represented as (l, m, u).

    r : float, default=0.5
        Shape parameter controlling the contribution of the left and right
        membership functions.

    Returns
    -------
    float
        Crisp distance value.

    Raises
    ------
    ValueError
        If inputs cannot be converted to valid TFNs or ``r`` is not numeric.
    RuntimeError
        If the distance calculation fails.

    """

    Validator.validate_tfn(a, 'a')
    Validator.validate_tfn(b, 'b')

    try:
        return float(
            (a[1] - b[1]) ** 2
            + ((a[1] - r * a[0]) -
               (b[1] - r * b[0])) ** 2
            + ((a[1] + r * a[2]) -
               (b[1] + r * b[2])) ** 2
        )

    except Exception as e:
        raise RuntimeError(f"Failed to compute L-R distance: {e}") from e


def tran_duckstein_distance(
    a: np.ndarray | list,
    b: np.ndarray | list
) -> float:
    """
    Calculate the Tran-Duckstein distance between two
    Triangular Fuzzy Numbers.

    The Tran-Duckstein distance combines differences between modal values,
    spreads, and covariance-like terms describing the shape of fuzzy
    numbers.

    Parameters
    ----------
    a : np.ndarray | list
        First Triangular Fuzzy Number represented as (l, m, u).

    b : np.ndarray | list
        Second Triangular Fuzzy Number represented as (l, m, u).

    Returns
    -------
    float
        Crisp distance value.

    Raises
    ------
    ValueError
        If inputs are not valid TFNs.
    RuntimeError
        If the calculation fails.
    """

    Validator.validate_tfn(a, 'a')
    Validator.validate_tfn(b, 'b')

    try:
        result = (
            (a[1] - b[1]) ** 2
            + 0.5 * (a[1] - b[1])
            * ((a[2] - a[0]) - (b[2] - b[0]))
            + 1 / 9
            * (
                (a[2] - a[1]) ** 2
                + (a[1] - a[0]) ** 2
                + (b[2] - b[1]) ** 2
                + (b[1] - b[0]) ** 2
            )
            - 1 / 9
            * (
                (a[1] - a[0]) * (a[2] - a[1])
                + (b[1] - b[0]) * (b[2] - b[1])
            )
            + 1 / 6
            * (
                2 * a[1] - a[0] - a[2]
            )
            * (
                2 * b[1] - b[0] - b[2]
            )
        )

        return float(result)

    except Exception as e:
        raise RuntimeError(f"Failed to compute Tran-Duckstein distance: {e}") from e

def mahdavi_distance(
    a: np.ndarray | list,
    b: np.ndarray | list
) -> float:
    """
    Calculate the Mahdavi distance between two Triangular Fuzzy Numbers.

    The Mahdavi distance is based on the Euclidean difference between
    corresponding TFN components and additionally considers the interaction
    between adjacent spreads.

    Parameters
    ----------
    a : np.ndarray | list
        First Triangular Fuzzy Number represented as (l, m, u).

    b : np.ndarray | list
        Second Triangular Fuzzy Number represented as (l, m, u).

    Returns
    -------
    float
        Crisp distance value.

    Raises
    ------
    ValueError
        If inputs cannot be converted to valid TFNs or do not contain
        exactly three components.

    RuntimeError
        If the distance calculation fails.
    """

    Validator.validate_tfn(a, 'a')
    Validator.validate_tfn(b, 'b')

    try:
        result = np.sqrt(
            1 / 6
            * (
                np.sum((b - a) ** 2)
                + (b[1] - a[1]) ** 2
                + np.sum(
                    [
                        (b[i] - a[i]) *
                        (b[i + 1] - a[i + 1])
                        for i in range(2)
                    ]
                )
            )
        )

        return float(result)

    except Exception as e:
        raise RuntimeError(f"Failed to compute Mahdavi distance: {e}") from e