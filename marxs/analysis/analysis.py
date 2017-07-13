# Licensed under GPL version 3 - see LICENSE.rst
from __future__ import division

import numpy as np
import scipy.optimize
from astropy.stats import sigma_clipped_stats

from ..optics import FlatDetector

def sigma_clipped_std(data, **kwargs):
    '''Return stddev of sigma-clipped data.

    Parameters
    ----------
    data : np.array
        unbinned input data
    kwargs : see `astropy.stats.sigma_clipped_stats`
        All keyword arguments are passed to `astropy.stats.sigma_clipped_stats`.
    '''
    mean, median, std = sigma_clipped_stats(data, **kwargs)
    return std


def find_best_detector_position(photons, col='det_x',
                                objective_func=sigma_clipped_std,
                                orientation=np.eye(3), **kwargs):
    '''Numerically find the position of best focus.

    This routine places detectors at different positions and
    calculates the width of the photon distribution for each position.
    As a side effect, ``photons`` will be set to the intersection with
    the last try of the detector position.

    Parameters
    ----------
    photons : `astropy.table.Table`
        Input photon list.
    col : string
        Column name of the photon distribution to be minimized.
        The default is set for detectors that look for a grating signal
        (which is dispersed in ``det_y`` direction).
    objective_func : function
        Function that accepts a np.array as input and return the width.
    rotation : np.array of shape (3,3)
        Rotation matrix for the detector. By default the detector is parallel
        to the yz plane of the global coordinate system (see `pos4d`).
    kwargs : see `scipy.optimize.minimize`
        All other keyword argument will be passed to `scipy.optimize.minimize`.
    Returns
    -------
    opt : OptimizeResult
        see `scipy.optimize.minimize`

    '''
    def width(x, photons):
        mdet = FlatDetector(position=np.array([x, 0, 0]), orientation=orientation, zoom=1e5, pixsize=1.)
        photons = mdet(photons)
        return objective_func(photons[col].data)

    return scipy.optimize.minimize(width, 0, args=(photons,), options={'maxiter': 20, 'disp': True},
                                   **kwargs)


def detected_fraction(photons, labels, col='order'):
    '''Calculate the fraction of photons detected for some integer label

    While written for calculating Aeff per order, this can be used with any discrete
    quantity, e.g. Aeff per CCD.

    Parameters
    ----------
    photons : `astropy.table.Table`
        Photon event list
    labels : np.array
        Numeric (integer) labels that are found in column ``col``. When, e.g.,
        the effective area per order is calculated, this array should contain
        the order numbers for which the calculation will be done.
    col : string
        Column name for the order column.

    Returns
    -------
    prop : np.array
        Probability for a photon in a specific order to be detected.
    '''
    prob = np.zeros_like(labels, dtype=float)
    for i, o in enumerate(labels):
        ind = (photons[col] == o)
        if filterfunc is not None:
            ind = ind & filterfunc(photons)
        prob[i] = np.sum(photons['probability'][ind]) / len(photons)
    return prob
