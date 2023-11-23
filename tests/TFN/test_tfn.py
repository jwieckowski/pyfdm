# Copyright (c) 2023 Jakub Więckowski

from pyfdm.TFN import TFN

def test_mathematical_operations():
    """
        Test verifying correctness of Triangular Fuzzy Numbers arithmetical operations.
        Reference value: Sudha, T., & Jayalalitha, G. (2020, July). Fuzzy triangular numbers in-Sierpinski triangle and right angle triangle. In Journal of Physics: Conference Series (Vol. 1597, No. 1, p. 012022). IOP Publishing.
    """

    tfn1 = TFN(1, 2, 3)
    tfn2 = TFN(2, 3, 4)

    # Test addition
    result_addition = tfn1 + tfn2
    assert result_addition == TFN(3, 5, 7)

    # Test subtraction
    result_subtraction = tfn1 - tfn2
    assert result_subtraction == TFN(-3, -1, 1)

    # Test multiplication
    result_multiplication = tfn1 * tfn2
    assert result_multiplication == TFN(2, 6, 12)

    # Test division
    result_division = tfn1 / tfn2
    assert result_division == TFN(0.25, 2/3, 1.5)

def test_membership_function():
    """
        Test verifying correctness of membership function calculation of Triangular Fuzzy Numbers.
        Reference value: Sudha, T., & Jayalalitha, G. (2020, July). Fuzzy triangular numbers in-Sierpinski triangle and right angle triangle. In Journal of Physics: Conference Series (Vol. 1597, No. 1, p. 012022). IOP Publishing.
    """
    

    tfn = TFN(1, 2, 3)

    # Test membership function at x_value
    x_value = 2.5
    membership_value = tfn.membership_function(x_value)
    assert membership_value == 0.5

def test_centroid():
    """
        Test verifying correctness of centroid calculation of Triangular Fuzzy Numbers.
        Reference value: Sudha, T., & Jayalalitha, G. (2020, July). Fuzzy triangular numbers in-Sierpinski triangle and right angle triangle. In Journal of Physics: Conference Series (Vol. 1597, No. 1, p. 012022). IOP Publishing.
    """
    tfn = TFN(1, 2, 3)

    # Test centroid
    centroid_value = tfn.centroid()
    assert centroid_value == 2

def test_core():
    """
        Test verifying correctness of core calculations of Triangular Fuzzy Numbers arithmetical.
        Reference value: Sudha, T., & Jayalalitha, G. (2020, July). Fuzzy triangular numbers in-Sierpinski triangle and right angle triangle. In Journal of Physics: Conference Series (Vol. 1597, No. 1, p. 012022). IOP Publishing.
    """
    tfn = TFN(1, 2, 3)

    # Test core
    core_values = tfn.core()
    assert core_values == [2]

def test_equality():
    """
        Test verifying correctness of equality of Triangular Fuzzy Numbers.
        Reference value: Sudha, T., & Jayalalitha, G. (2020, July). Fuzzy triangular numbers in-Sierpinski triangle and right angle triangle. In Journal of Physics: Conference Series (Vol. 1597, No. 1, p. 012022). IOP Publishing.
    """
    tfn1 = TFN(1, 2, 3)
    tfn2 = TFN(1, 2, 3)

    # Test equality
    assert tfn1 == tfn2

def test_inclusion():
    """
        Test verifying correctness of inclusion of Triangular Fuzzy Numbers.
        Reference value: Sudha, T., & Jayalalitha, G. (2020, July). Fuzzy triangular numbers in-Sierpinski triangle and right angle triangle. In Journal of Physics: Conference Series (Vol. 1597, No. 1, p. 012022). IOP Publishing.
    """
    tfn1 = TFN(1, 2, 3)
    tfn2 = TFN(0, 1, 4)

    # Test inclusion
    assert tfn1.is_included_in(tfn2)


def test_s_norm():
    """
        Test verifying correctness of S-norm operator of Triangular Fuzzy Numbers.
        Reference value: Sudha, T., & Jayalalitha, G. (2020, July). Fuzzy triangular numbers in-Sierpinski triangle and right angle triangle. In Journal of Physics: Conference Series (Vol. 1597, No. 1, p. 012022). IOP Publishing.
    """
    tfn1 = TFN(1, 2, 3)
    tfn2 = TFN(2, 3, 4)

    # Test fuzzy OR operation (S-norm)
    result_s_norm = tfn1.s_norm(tfn2)

    # Expected result
    expected_result = TFN(2, 3, 4)

    # Check if the result matches the expected result
    assert result_s_norm == expected_result

def test_t_norm():
    """
        Test verifying correctness of T-norm operator Triangular Fuzzy Numbers.
        Reference value: Sudha, T., & Jayalalitha, G. (2020, July). Fuzzy triangular numbers in-Sierpinski triangle and right angle triangle. In Journal of Physics: Conference Series (Vol. 1597, No. 1, p. 012022). IOP Publishing.
    """
    tfn1 = TFN(1, 2, 3)
    tfn2 = TFN(2, 3, 4)

    # Test fuzzy AND operation (T-norm)
    result_t_norm = tfn1.t_norm(tfn2)

    # Expected result
    expected_result = TFN(1, 2, 3)

    # Check if the result matches the expected result
    assert result_t_norm == expected_result

