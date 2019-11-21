"""
Plotting functions: scatter, plot (curves), axis labelling.
"""

import matplotlib
from matplotlib import pyplot
import numpy as np
import math

from .helpers import project_sequence
from .colormapping import get_cmap, colorbar_hack


### Drawing Helpers ###

def resize_drawing_canvas(ax, scale=1.):
    """
    Makes sure the drawing surface is large enough to display projected
    content.

    Parameters
    ----------
    ax: Matplotlib AxesSubplot, None
        The subplot to draw on.
    scale: float, 1.0
        Simplex scale size.
    """
    ax.set_ylim((-0.10 * scale, .90 * scale))
    ax.set_xlim((-0.05 * scale, 1.05 * scale))


def clear_matplotlib_ticks(ax=None, axis="both"):
    """
    Clears the default matplotlib axes, or the one specified by the axis
    argument.

    Parameters
    ----------
    ax: Matplotlib AxesSubplot, None
        The subplot to draw on.
    axis: string, "both"
        The axis to clear: "x" or "horizontal", "y" or "vertical", or "both"
    """
    if not ax:
        return
    if axis.lower() in ["both", "x", "horizontal"]:
        ax.set_xticks([], [])
    if axis.lower() in ["both", "y", "vertical"]:
        ax.set_yticks([], [])


## Curve Plotting ##

def plot(points, ax=None, permutation=None, **kwargs):
    """
    Analogous to maplotlib.plot. Plots trajectory points where each point is a
    tuple (x,y,z) satisfying x + y + z = scale (not checked). The tuples are
    projected and plotted as a curve.

    Parameters
    ----------
    points: List of 3-tuples
        The list of tuples to be plotted as a connected curve.
    ax: Matplotlib AxesSubplot, None
        The subplot to draw on.
    kwargs:
        Any kwargs to pass through to matplotlib.
    """
    if not ax:
        fig, ax = pyplot.subplots()
    xs, ys = project_sequence(points, permutation=permutation)
    ax.plot(xs, ys, **kwargs)
    return ax

def arrow(points, ax=None, permutation=None, arrows=1, start=False, end=False, **kwargs):
    """
    Analogous to maplotlib.arrow. Plots trajectory points where each point is a
    tuple (x,y,z) satisfying x + y + z = scale (not checked). The tuples are
    projected and plotted as a curve.

    Parameters
    ----------
    points: List of 3-tuples
        The list of tuples to be plotted as a connected curve.
    ax: Matplotlib AxesSubplot, None
        The subplot to draw on.
    kwargs:
        Any kwargs to pass through to matplotlib.
    """
    if not ax:
        fig, ax = pyplot.subplots()
    xs, ys = project_sequence(points, permutation=permutation)
    # plots an arrow at the 1/(arrows+1) intervals excluding 0 and 1
    # so arrows = 1 puts 1 arrow halfway, 2 puts an arrow at 1/3 and 2/3
    interval = int(np.floor(len(xs)/(arrows+1)))
    # if start then plot an arrow on the first segment
    if start:
        x,y = xs[0], ys[0]
        dx,dy = xs[1]-x, ys[1]-y
        ax.arrow(x,y,dx,dy, **kwargs)
    # if endthen plot an arrow on the last segment
    if end:
        x,y = xs[-1], ys[-1]
        dx,dy = xs[-1]-xs[-2], ys[-1]-ys[-2]
        ax.arrow(x,y,dx,dy, **kwargs)
    
    for i in range(arrows):
        # plot an arrow from x_i,y_i to x_i+1,y_i+1
        x,y = xs[(i+1)*interval],ys[(i+1)*interval]
        dx,dy = xs[(i+1)*interval+1]-x, ys[(i+1)*interval+1]-y
        ax.arrow(x,y,dx,dy, **kwargs)
    return ax

def plot_colored_trajectory(points, ax=None, permutation=None,
                            **kwargs):
    """
    Plots trajectories with changing color, simlar to `plot`. Trajectory points
    are tuples (x,y,z) satisfying x + y + z = scale (not checked). The tuples are
    projected and plotted as a curve.

    Parameters
    ----------
    points: List of 3-tuples
        The list of tuples to be plotted as a connected curve.
    ax: Matplotlib AxesSubplot, None
        The subplot to draw on.
    cmap: String or matplotlib.colors.Colormap, None
        The name of the Matplotlib colormap to use.
    kwargs:
        Any kwargs to pass through to matplotlib.
    """
    if not ax:
        fig, ax = pyplot.subplots()
    xs, ys = project_sequence(points, permutation=permutation)

    segments = []
    for i in range(len(xs) - 1):
        cur_line = []
        x_before = xs[i]
        y_before = ys[i]
        x_after = xs[i+1]
        y_after = ys[i+1]

        cur_line.append([x_before, y_before])
        cur_line.append([x_after, y_after])
        segments.append(cur_line)
    segments = np.array(segments)
                       
    line_segments = matplotlib.collections.LineCollection(segments, **kwargs)
    line_segments.set_array(np.arange(len(segments)))
    ax.add_collection(line_segments)

    return ax

def scatter(points, ax=None, permutation=None, colorbar=False, colormap=None,
            vmin=0, vmax=1, scientific=False, cbarlabel=None, cb_kwargs=None,
            **kwargs):
    """
    Plots trajectory points where each point satisfies x + y + z = scale.
    First argument is a list or numpy array of tuples of length 3.

    Parameters
    ----------
    points: List of 3-tuples
        The list of tuples to be scatter-plotted.
    ax: Matplotlib AxesSubplot, None
        The subplot to draw on.
    colorbar: bool, False
        Show colorbar.
    colormap: String or matplotlib.colors.Colormap, None
        The name of the Matplotlib colormap to use.
    vmin: int, 0
        Minimum value for colorbar.
    vmax: int, 1
        Maximum value for colorbar.
    cb_kwargs: dict
        Any additional kwargs to pass to colorbar
    kwargs:
        Any kwargs to pass through to matplotlib.
    """
    if not ax:
        fig, ax = pyplot.subplots()
    xs, ys = project_sequence(points, permutation=permutation)
    ax.scatter(xs, ys, vmin=vmin, vmax=vmax, **kwargs)

    if colorbar and (colormap != None):
        if cb_kwargs != None:
            colorbar_hack(ax, vmin, vmax, colormap, scientific=scientific,
                          cbarlabel=cbarlabel, **cb_kwargs)
        else:
            colorbar_hack(ax, vmin, vmax, colormap, scientific=scientific,
                          cbarlabel=cbarlabel)

    return ax
