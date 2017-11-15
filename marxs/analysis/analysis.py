# Licensed under GPL version 3 - see LICENSE.rst
from __future__ import division

import numpy as np
import scipy.optimize
from astropy.stats import sigma_clipped_stats

from ..optics import FlatDetector

def sigma_clipped_std(photons, colname='det_x', **kwargs):
    '''Return stddev of sigma-clipped data.

    Parameters
    ----------
    photons : `astropy.table.Table`
        Input photon list.
    colname : string
        Column name of the photon distribution to be minimized.
        The default is set for detectors that look for a grating signal
        (which is dispersed in ``det_y`` direction).
    kwargs : see `astropy.stats.sigma_clipped_stats`
        All keyword arguments are passed to `astropy.stats.sigma_clipped_stats`.
    '''
    mean, median, std = sigma_clipped_stats(photons[colname].data, **kwargs)
    return std


def mean_width_2d(photons):
    '''Return average distance from center of det_x, det_y distribution.

    Parameters
    ----------
    photons : `astropy.table.Table`
        Input photon list.
    '''
    x = photons['det_x'].data
    y = photons['det_y'].data
    r = np.linalg.norm(np.stack([x - x.mean(), y - y.mean()]), axis=0)
    return np.sum(r) / len(r)



def find_best_detector_position(photons,
                                objective_func=sigma_clipped_std,
                                objective_func_args={'colname': 'det_x'},
                                orientation=np.eye(3), **kwargs):
    '''Numerically find the position of best focus.

    This routine places detectors at different positions and
    calculates the width of the photon distribution for each position.
    The detector is moved perpendicular to its plane.
    As a side effect, ``photons`` will be set to the intersection with
    the last try of the detector position.

    Parameters
    ----------
    photons : `astropy.table.Table`
        Input photon list.
    objective_func : function
        Function that accepts a np.array as input and return the width.
    object_func_args : dict
        Dict of any other options for the `objective_func`.
    rotation : np.array of shape (3,3)
        Rotation matrix for the detector. By default the detector is parallel
        to the yz plane of the global coordinate system (see `pos4d`).
    kwargs : see `scipy.optimize.minimize`
        All other keyword argument will be passed to `scipy.optimize.minimize`.
    Returns
    -------
    opt : OptimizeResult
        see `scipy.optimize.minimize_scalar`

    '''
    def width(x, photons):
        mdet = FlatDetector(position=np.dot(orientation, np.array([x, 0, 0])),
                            orientation=orientation,
                            zoom=1e5, pixsize=1.)
        photons = mdet(photons)
        return objective_func(photons, **objective_func_args)

    return scipy.optimize.minimize_scalar(width, args=(photons,), options={'maxiter': 20},
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
