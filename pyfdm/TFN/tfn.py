# Copyright (c) 2023 Jakub Więckowski
import numpy as np

class TFN:
    def __init__(self, a, b, c):
        """
        Initializes a Triangular Fuzzy Number with parameters a, b, and c.

        Parameters:
        - a: Lower bound
        - b: Peak (mode)
        - c: Upper bound
        """

        if not (a <= b <= c):
            raise ValueError(f'a should be less of equal to b, and b should be less or equal to c.')
        
        self.a = a
        self.b = b
        self.c = c

    def __repr__(self) -> str:
        """
        Returns a string representation of the Triangular Fuzzy Number.
        """

        return f"TFN({self.a}, {self.b}, {self.c})"

    def __str__(self):
        return f'({self.a}, {self.b}, {self.c})'


    def __add__(self, other) -> 'TFN':
        """
        Overloads the '+' operator for addition of Triangular Fuzzy Numbers.
        Also handling an addition with number.

        Returns a new Triangular Fuzzy Number representing the sum.
        """
        if isinstance(other, TFN):
            a = self.a + other.a
            b = self.b + other.b
            c = self.c + other.c

            return TFN(a, b, c)
        else:
            return TFN(self.a + other, self.b + other, self.c + other)

    def __sub__(self, other) -> 'TFN':
        """
        Overloads the '-' operator for subtraction of Triangular Fuzzy Numbers.
        Also handling a subtraction with number.

        Returns a new Triangular Fuzzy Number representing the difference.
        """

        if isinstance(other, TFN):
            a = self.a - other.c
            b = self.b - other.b
            c = self.c - other.a

            return TFN(a, b, c)
        else:
            return self + (- other)

    def __mul__(self, other) -> 'TFN':
        """
        Overloads the '*' operator for multiplication of Triangular Fuzzy Numbers.
        Also handling a multiplication by number.

        Returns a new triangular fuzzy number representing the product.
        """

        if isinstance(other, TFN):
            a = min(self.a * other.a, self.a * other.c, self.c * other.a, self.c * other.c)
            b = self.b * other.b
            c = max(self.a * other.a, self.a * other.c, self.c * other.a, self.c * other.c)

            return TFN(a, b, c)
        else:
            return TFN(self.a * other, self.b * other, self.c * other)

    def __truediv__(self, other: 'TFN') -> 'TFN':
        """
        Overloads the '/' operator for division of Triangular Fuzzy Numbers.
        Also handling a division by number.
        
        Returns a new Triangular Fuzzy Number representing the quotient.

        Raises a ValueError if the denominator contains zero.
        """

        if isinstance(other, TFN):
            if other.a <= 0 <= other.c:
                raise ValueError("Division by a Triangular Fuzzy Number containing zero is undefined.")
            
            a = min(self.a / other.a, self.a / other.c, self.c / other.a, self.c / other.c)
            b = self.b / other.b
            c = max(self.a / other.a, self.a / other.c, self.c / other.a, self.c / other.c)

            return TFN(a, b, c)
        else:
        
            return TFN(self.a / other, self.b / other, self.c / other)

    def __eq__(self, other: 'TFN') -> bool:
        """
        Checks if two Triangular Fuzzy Numbers are equal.
        Returns True if they are equal, False otherwise.
        """
        return self.a == other.a and self.b == other.b and self.c == other.c

    def __le__(self, other):
        return self.a <= other.a

    def __ge__(self, other):
        return self.c >= other.c

    def __abs__(self):
        a = abs(self.a)
        b = abs(self.b)
        c = abs(self.c)
        if a > c:
            return TFN(c, b, a)
        return TFN(a, b, c)

    def __round__(self, value):
        return TFN(round(self.a, value), round(self.b, value), round(self.c, value))

    def membership_function(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Calculates the membership function value at a given point x.
        Also handles calculation for an array of values.

        Parameters:
        - x (float): The point at which to calculate the membership function.

        Returns:
        - float: The membership function value at the given point x.
        """
        
        if isinstance(x, np.ndarray):
            return self._membership_array(x)
        else:
            return self._membership_number(x)

    def _membership_array(self, x):
        res = np.zeros(x.shape)
        mask = x == self.b
        res[mask] = 1

        mask = np.logical_and(x > self.a, x < self.b)
        res[mask] = (x[mask] - self.a) / (self.b - self.a)

        mask = np.logical_and(x < self.c, x > self.b)
        res[mask] = (self.c - x[mask]) / (self.c - self.b)
        return res

    def _membership_number(self, x):
        if x <= self.a or x >= self.c:
            return 0.0
        elif self.a < x <= self.b:
            return (x - self.a) / (self.b - self.a)
        elif self.b <= x < self.c:
            return (self.c - x) / (self.c - self.b)

    def centroid(self) -> float:
        """
        Calculates the centroid of the Triangular Fuzzy Number.
        
        Returns:
        - float: The centroid of the Triangular Fuzzy Number.
        """

        return (self.a + self.b + self.c) / 3

    def core(self) -> list:
        """
        Calculates the core of the Triangular Fuzzy Number.
        
        Returns:
        - list: A list containing the core values of the Triangular Fuzzy Number.
        """

        return [self.b]

    def is_included_in(self, other: 'TFN') -> bool:
        """
        Checks if the current Triangular Fuzzy Number is included in the other.
        
        Returns:
        - bool: True if the current Triangular Fuzzy Number is included in the other, False otherwise.
        """
        return self.a >= other.a and self.c <= other.c

    def s_norm(self, other):
        """
        S-norm operator for fuzzy OR operation.

        Parameters:
        - other (TFN): Another Triangular Fuzzy Number.

        Returns:
        - TFN: Result of the fuzzy OR operation.
        """
        a = max(self.a, other.a)
        b = max(self.b, other.b)
        c = max(self.c, other.c)
        return TFN(a, b, c)

    def t_norm(self, other):
        """
        T-norm operator for fuzzy AND operation.

        Parameters:
        - other (TFN): Another Triangular Fuzzy Number.

        Returns:
        - TFN: Result of the fuzzy AND operation.
        """
        a = min(self.a, other.a)
        b = min(self.b, other.b)
        c = min(self.c, other.c)
        return TFN(a, b, c)

    
