# Copyright (c) 2023 Jakub Więckowski

import matplotlib.pyplot as plt
import numpy as np
from ..TFN import TFN

def tfn_membership_plot(a, b, c, x, title=None, ax=None):
    """
        Visualize the membership function of a Triangular Fuzzy Number (TFN) at a specified x value.

        Parameters:
        - a (float): The leftmost point of the triangular fuzzy number.
        - b (float): The peak or center point of the triangular fuzzy number.
        - c (float): The rightmost point of the triangular fuzzy number.
        - x (float): The specific x value at which to highlight the membership function.
        - title (str, optional): The title of the plot. If not provided, the default title is 'Membership of TFN'.
        - ax (Axes or None): Axes object to draw on. If None, then current axes is used.

        Example:
        ```
        # Example Usage:
        tfn_membership_plot(2, 4, 6, 3.5, title='Membership Visualization')
        ```

        This function generates a plot illustrating the membership function of a Triangular Fuzzy Number (TFN) defined by parameters a, b, and c. The specified x value is marked with a circle on the plot, and its corresponding membership value is indicated by a dashed line. Vertical dashed lines represent the TFN's parameters (a, b, c).

        The title of the plot can be specified using the optional `title` parameter. If no title is provided, the default title is 'Membership of TFN'.

        The function uses Matplotlib for plotting, and the resulting plot is displayed.
    """

    if ax is None:
        ax = plt.gca()

    tfn = TFN(a, b, c)

    # Generate x values for the plot
    x_values = np.linspace(tfn.a - 1, tfn.c + 1, 1000)

    # Calculate y values for the membership function
    y_values = [tfn.membership_function(x) for x in x_values]

    # Plot the triangular fuzzy number with membership function values
    ax.plot(x_values, y_values)
    ax.fill_between(x_values, 0, y_values, color='lightgray', alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(tfn.a, linestyle='--', color='red', label='a')
    ax.axvline(tfn.b, linestyle='--', color='green', label='b')
    ax.axvline(tfn.c, linestyle='--', color='blue', label='c')

    # Mark the specified x value with a circle
    highlighted_membership_value = tfn.membership_function(x)
    ax.scatter(x, highlighted_membership_value, color='orange', zorder=5)
    ax.axvline(x, linestyle='--', color='orange', label='$\mu$(x)')
    ax.annotate(f'$\mu$={x}', (x, highlighted_membership_value), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)

    # Add labels and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylabel('Membership Value $\mu$(x)')
    title = title if title is not None else 'Membership of TFN' 
    ax.set_title(title)
    ax.legend()

    return ax