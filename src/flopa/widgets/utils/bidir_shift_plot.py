# flopa/widgets/utils/bidir_shift_plot.py

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np


def plot_bidirectional_shift(
    plot_data: np.ndarray,
    ax: Axes = None
) -> Figure:
    """
    Plots the results of a bidirectional shift estimation.

    Args:
        plot_data (np.ndarray): The 2D numpy array returned by 
                                estimate_bidirectional_shift containing
                                shifts, scores, and the Gaussian fit.
        ax (matplotlib.axes.Axes, optional): An existing Axes object to plot on.
                                             If None, a new figure and axes are created.

    Returns:
        matplotlib.figure.Figure: The figure containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    shifts, scores, fit = plot_data[0], plot_data[1], plot_data[2]
    
    scores_normalized = (scores - scores.min()) / (scores.max() - scores.min())
    fit_normalized = (fit - fit.min()) / (fit.max() - fit.min())
    
    ax.plot(shifts, scores_normalized, 'o', label='Correlation Score', markersize=5)
    ax.plot(shifts, fit_normalized, '-', label='Gaussian Fit', linewidth=2)
    
    peak_shift = shifts[np.argmax(fit)]
    ax.axvline(peak_shift, color='r', linestyle='--', label=f'Peak: {peak_shift:.4f}')
    
    ax.set_title("Bidirectional Shift Estimation")
    ax.set_xlabel("Phase Shift")
    ax.set_ylabel("Normalized Correlation Score")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    
    return fig