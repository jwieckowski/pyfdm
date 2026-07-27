# Copyright (c) 2023 - 2026 Jakub Więckowski

from __future__ import annotations
import numpy as np

__all__ = ["TFN"]

class TFN:
    """
    Representation of a Triangular Fuzzy Number.

    A Triangular Fuzzy Number (TFN) is defined by three parameters:

    .. math::

        A = (a, b, c)

    where:

    - ``a`` is the lower bound,
    - ``b`` is the modal value,
    - ``c`` is the upper bound.

    Parameters
    ----------
    a : int | float
        Lower bound of the TFN.
    b : int | float
        Modal value (membership degree equal to 1).
    c : int | float
        Upper bound of the TFN.

    Raises
    ------
    ValueError
        If the TFN does not satisfy ``a <= b <= c``.
    TypeError
        If input values are not numeric.
    """

    def __init__(
        self,
        a: int | float,
        b: int | float,
        c: int | float
    ) -> None:

        try:
            if not all(
                isinstance(x, (int, float))
                for x in (a, b, c)
            ):
                raise TypeError("TFN parameters must be numeric values.")

            if not (a <= b <= c):
                raise ValueError( "TFN parameters must satisfy a <= b <= c.")

            self.a = float(a)
            self.b = float(b)
            self.c = float(c)

        except (TypeError, ValueError):
            raise

        except Exception as e:
            raise RuntimeError(f"Failed to initialize TFN: {e}") from e


    def __repr__(self) -> str:
        """
        Return developer-oriented string representation.

        Returns
        -------
        str
            TFN representation.
        """

        return f"TFN({self.a}, {self.b}, {self.c})"

    def __str__(self) -> str:
        """
        Return human-readable TFN representation.

        Returns
        -------
        str
            TFN values formatted as tuple.
        """

        return f"({self.a}, {self.b}, {self.c})"

    def _validate_other(self, other: TFN) -> None:
        """
        Validate another TFN instance.

        Parameters
        ----------
        other : TFN
            Object to validate.

        Raises
        ------
        TypeError
            If object is not a TFN.
        """

        if not isinstance(other, TFN):
            raise TypeError(f"Expected TFN object, got {type(other).__name__}.")

    def __add__(self, other: TFN | int | float) -> TFN:
        """
        Add two TFNs or add a scalar value.

        Parameters
        ----------
        other : TFN | int | float
            TFN or scalar value.

        Returns
        -------
        TFN
            Resulting TFN.

        Raises
        ------
        TypeError
            If the operand type is unsupported.
        """

        try:
            if isinstance(other, TFN):
                return TFN(
                    self.a + other.a,
                    self.b + other.b,
                    self.c + other.c
                )

            if isinstance(other, (int, float)):
                return TFN(
                    self.a + other,
                    self.b + other,
                    self.c + other
                )

            return NotImplemented

        except Exception as e:
            raise RuntimeError(f"TFN addition failed: {e}") from e


    def __sub__(self, other: TFN | int | float) -> TFN:
        """
        Subtract another TFN or scalar value.

        Parameters
        ----------
        other : TFN | int | float
            TFN or scalar value.

        Returns
        -------
        TFN
            Resulting TFN.

        Raises
        ------
        TypeError
            If the operand type is unsupported.
        """

        try:

            if isinstance(other, TFN):
                return TFN(
                    self.a - other.c,
                    self.b - other.b,
                    self.c - other.a
                )

            if isinstance(other, (int, float)):
                return TFN(
                    self.a - other,
                    self.b - other,
                    self.c - other
                )

            return NotImplemented

        except Exception as e:
            raise RuntimeError(f"TFN subtraction failed: {e}") from e

    def __mul__(self, other: TFN | int | float) -> TFN:
        """
        Multiply two TFNs or multiply a TFN by a scalar.

        Parameters
        ----------
        other : TFN | int | float
            TFN or scalar value.

        Returns
        -------
        TFN
            Resulting TFN.

        Raises
        ------
        TypeError
            If the operand type is unsupported.
        """

        try:

            if isinstance(other, TFN):
                values = [
                    self.a * other.a,
                    self.a * other.c,
                    self.c * other.a,
                    self.c * other.c
                ]

                return TFN(
                    min(values),
                    self.b * other.b,
                    max(values)
                )

            if isinstance(other, (int, float)):
                values = [
                    self.a * other,
                    self.b * other,
                    self.c * other
                ]

                return TFN(
                    min(values),
                    values[1],
                    max(values)
                )

            return NotImplemented

        except Exception as e:
            raise RuntimeError(f"TFN multiplication failed: {e}") from e

    def __truediv__(self, other: TFN | int | float) -> TFN:
        """
        Divide a TFN by another TFN or scalar.

        For TFN division, the resulting bounds are calculated using
        the extension principle.

        Parameters
        ----------
        other : TFN | int | float
            Divisor TFN or scalar value.

        Returns
        -------
        TFN
            Resulting TFN.

        Raises
        ------
        ValueError
            If the divisor contains zero.
        TypeError
            If the operand type is unsupported.
        RuntimeError
            If division cannot be computed.
        """

        try:

            if isinstance(other, TFN):
                if other.a <= 0 <= other.c:
                    raise ValueError("Division by TFN containing zero is undefined.")

                values = [
                    self.a / other.a,
                    self.a / other.c,
                    self.c / other.a,
                    self.c / other.c
                ]

                return TFN(
                    min(values),
                    self.b / other.b,
                    max(values)
                )

            if isinstance(other, (int, float)):
                if other == 0:
                    raise ValueError("Division by zero is undefined.")

                values = [
                    self.a / other,
                    self.b / other,
                    self.c / other
                ]

                return TFN(
                    min(values),
                    values[1],
                    max(values)
                )

            return NotImplemented

        except (ValueError, TypeError):
            raise

        except Exception as e:
            raise RuntimeError(f"TFN division failed: {e}") from e

    def __eq__(self, other: TFN) -> bool:
        """
        Check equality between two TFNs.

        Parameters
        ----------
        other : TFN
            TFN to compare.

        Returns
        -------
        bool
            True if both TFNs have identical parameters.
        """

        if not isinstance(other, TFN):
            return False

        return (
            self.a == other.a
            and self.b == other.b
            and self.c == other.c
        )

    def __le__(self, other: TFN) -> bool:
        """
        Compare TFNs using lower bounds.

        Parameters
        ----------
        other : TFN
            TFN to compare.

        Returns
        -------
        bool
            True if this TFN lower bound is smaller or equal.

        Raises
        ------
        TypeError
            If other object is not a TFN.
        """

        self._validate_other(other)
        return self.a <= other.a

    def __ge__(self, other: TFN) -> bool:
        """
        Compare TFNs using upper bounds.

        Parameters
        ----------
        other : TFN
            TFN to compare.

        Returns
        -------
        bool
            True if this TFN upper bound is greater or equal.

        Raises
        ------
        TypeError
            If other object is not a TFN.
        """

        self._validate_other(other)

        return self.c >= other.c

    def __abs__(self) -> TFN:
        """
        Return absolute value of a TFN.

        Returns
        -------
        TFN
            Absolute TFN.
        """

        try:

            values = [
                abs(self.a),
                abs(self.b),
                abs(self.c)
            ]

            return TFN(
                min(values),
                values[1],
                max(values)
            )

        except Exception as e:
            raise RuntimeError(f"Absolute TFN operation failed: {e}") from e

    def __round__(self, ndigits: int = 0) -> TFN:
        """
        Round TFN values.

        Parameters
        ----------
        ndigits : int
            Number of decimal digits.

        Returns
        -------
        TFN
            Rounded TFN.
        """

        try:
            return TFN(
                round(self.a, ndigits),
                round(self.b, ndigits),
                round(self.c, ndigits)
            )

        except Exception as e:
            raise RuntimeError(f"TFN rounding failed: {e}") from e

    def membership_function(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Calculate the membership degree of a value.

        The membership function of a TFN is defined as:

        .. math::

            \\mu_A(x)=
            \\begin{cases}
            0, & x \\le a \\\\
            \\frac{x-a}{b-a}, & a < x \\le b \\\\
            \\frac{c-x}{c-b}, & b < x < c \\\\
            0, & x \\ge c
            \\end{cases}

        Parameters
        ----------
        x : float | np.ndarray
            Input value or array of values.

        Returns
        -------
        float | np.ndarray
            Membership degree(s) in interval [0, 1].

        Raises
        ------
        TypeError
            If input type is unsupported.

        RuntimeError
            If membership calculation fails.
        """

        try:

            if isinstance(x, np.ndarray):
                return self._membership_array(x)

            if isinstance(x, (int, float)):
                return self._membership_number(float(x))

            raise TypeError(f"'x' must be float or np.ndarray, got {type(x).__name__}.")

        except (TypeError, ValueError):
            raise

        except Exception as e:
            raise RuntimeError(f"Membership function calculation failed: {e}") from e

    def _membership_array(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate membership degrees for an array of values.

        Parameters
        ----------
        x : np.ndarray
            Array of input values.

        Returns
        -------
        np.ndarray
            Membership values corresponding to ``x``.

        Raises
        ------
        RuntimeError
            If vectorized computation fails.
        """

        try:

            x = np.asarray(x, dtype=float)
            result = np.zeros(x.shape, dtype=float)

            # maximum membership point
            mask = x == self.b
            result[mask] = 1.0

            # increasing part
            if self.b != self.a:
                mask = (
                    (x > self.a)
                    &
                    (x < self.b)
                )

                result[mask] = (
                    (x[mask] - self.a)
                    /
                    (self.b - self.a)
                )

            # decreasing part
            if self.c != self.b:
                mask = (
                    (x > self.b)
                    &
                    (x < self.c)
                )

                result[mask] = (
                    (self.c - x[mask])
                    /
                    (self.c - self.b)
                )

            return result

        except Exception as e:
            raise RuntimeError(f"Array membership calculation failed: {e}") from e

    def _membership_number(self, x: int | float) -> float:
        """
        Calculate membership degree for a single value.

        Parameters
        ----------
        x : int | float
            Input value.

        Returns
        -------
        float
            Membership degree.

        Raises
        ------
        RuntimeError
            If calculation fails.
        """

        try:
            x = float(x)

            if x <= self.a or x >= self.c:
                return 0.0

            if x == self.b:
                return 1.0

            if self.a < x < self.b:
                if self.b == self.a:
                    return 1.0

                return (x - self.a) / (self.b - self.a)

            if self.b < x < self.c:
                if self.c == self.b:
                    return 1.0

                return (self.c - x) / (self.c - self.b)

            return 0.0

        except Exception as e:
            raise RuntimeError(f"Scalar membership calculation failed: {e}") from e
