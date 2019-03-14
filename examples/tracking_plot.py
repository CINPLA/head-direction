import neo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import simpleaxis
import math
import matplotlib.gridspec as gridspec


def plot_head_direction_rate(ang_bins, rate_in_ang, projection='polar',
                             ax=None, color='k'):
    """


    Parameters
    ----------
    ang_bins : angular binsize
    rate_in_ang :
    projection : 'polar' or None
    ax : matplotlib axes

    Returns
    -------
    out : ax
    """
    import math
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=projection)
    binsize = ang_bins[1] - ang_bins[0]
    if projection is None:
        ax.set_xlim(0, 2 * np.pi)
    elif projection == 'polar':
        ax.set_xticks([0, np.pi])
    ax.bar(ang_bins, rate_in_ang, width=binsize, color=color)
    return ax
