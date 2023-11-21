# Copyright (c) 2023 Jakub Więckowski

import numpy as np
import matplotlib.pyplot as plt
from ..TFN import TFN

def single_tfn_plot(a, b, c, title=None, ax=None):
    """
        Visualize a single Triangular Fuzzy Number (TFN) on a plot.

        Parameters:
        - a (float): The leftmost point of the triangular fuzzy number.
        - b (float): The peak or center point of the triangular fuzzy number.
        - c (float): The rightmost point of the triangular fuzzy number.
        - title (str, optional): The title of the plot. If not provided, the default title is 'Triangular Fuzzy Number'.
        - ax (Axes or None): Axes object to draw on. If None, then current axes is used.

        Example:
        ```
        # Example Usage:
        single_tfn_plot(2, 4, 6, title='Example Single TFN Plot')
        ```

        The `a`, `b`, and `c` parameters represent the leftmost, peak, and rightmost points of the Triangular Fuzzy Number (TFN). The function generates a plot with the TFN's membership function, filling the area under the curve with light gray for better visualization. Vertical dashed lines are added to indicate the TFN's parameters (a, b, c).

        The title of the plot can be specified using the optional `title` parameter. If no title is provided, the default title is 'Triangular Fuzzy Number'.

        The function uses Matplotlib for plotting, and the resulting plot is displayed.
    """

    if ax is None:
        ax = plt.gca()

    tfn = TFN(a, b, c)

    # Generate x values for the plot
    x_values = np.linspace(tfn.a - 1, tfn.c + 1, 1000)

    # Calculate y values using the membership function
    y_values = [tfn.membership_function(x) for x in x_values]

    # Plot the triangular fuzzy number
    ax.plot(x_values, y_values, label='TFN', linewidth=2)
    ax.fill_between(x_values, 0, y_values, color='lightgray', alpha=0.7)

    # Add grid and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(tfn.a, linestyle='--', color='red', label='a')
    ax.axvline(tfn.b, linestyle='--', color='green', label='b')
    ax.axvline(tfn.c, linestyle='--', color='blue', label='c')

    ax.set_ylabel('Membership Value $\mu$(x)')
    title = title if title is not None else 'Triangular Fuzzy Number' 
    ax.set_title(title)
    ax.legend(loc='upper right')

    return ax