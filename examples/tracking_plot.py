import neo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import quantities as pq
from exana.tracking.fields import (gridness, occupancy_map,
                     spatial_rate_map,
                     spatial_rate_map_1d)
from exana.tracking.head import *
from utils import simpleaxis
import math
from scipy.ndimage.measurements import center_of_mass
import matplotlib.gridspec as gridspec
from exana.misc.tools import is_quantities


def plot_head_direction_rate(ang_bins, rate_in_ang, projection='polar',
                             normalization=False, ax=None, color='k'):
    """


    Parameters
    ----------
    ang_bins : angular binsize
    rate_in_ang :
    projection : 'polar' or None
    normalization :
    group_name
    ax : matplotlib axes
    mask_unvisited : True: mask bins which has not been visited

    Returns
    -------
    out : ax
    """
    import math
    assert ang_bins.units == pq.degrees, 'ang_bins must be in degrees'
    if normalization:
        rate_in_ang = normalize(rate_in_ang, mode='minmax')
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=projection)
    binsize = ang_bins[1] - ang_bins[0]
    if projection is None:
        ax.set_xticks(range(0, 360 + 60, 60))
        ax.set_xlim(0, 360)
    elif projection == 'polar':
        ang_bins = [math.radians(deg) for deg in ang_bins] * pq.radians
        binsize = math.radians(binsize) * pq.radians
        ax.set_xticks([0, np.pi])
    ax.bar(ang_bins, rate_in_ang, width=binsize, color=color)
    return ax
