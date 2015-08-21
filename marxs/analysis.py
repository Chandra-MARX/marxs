import numpy as np
import scipy.optimize
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats

from .optics import FlatDetector

def measure_FWHM(data):
    '''Obtain the FWHM of some quantity in an eventlist.

    This function provides a robust measure of the FWHM of a quantity.
    It ignores outliers by first sigma clipping the input data, then it
    fits a Gaussian to the histogram of the sigma-clipped data.

    Even if the data is not Gaussian distributed, this should provide a good
    estimate of the real FWHM and it is a lot more stable then calculating the "raw" FWHM
    of the histrogram, which can depend sensitively on the bin size, if the max value is
    driven by noise.

    Parameters
    ----------
    data : np.array
        unbinned input data

    Results
    -------
    FWHM : float
        robust estimate for FWHM
    '''
    # Get an estimate of a sensible bin width for histogram
    mean, median, std = sigma_clipped_stats(data)
    n = len(data)
    hist, bin_edges = np.histogram(data, range=mean + np.array([-3, 3]) * std, bins = n/10)
    g_init = models.Gaussian1D(amplitude=hist.max(), mean=mean, stddev=std)
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, (bin_edges[:-1] + bin_edges[1:]) / 2., hist)
    return 2.3548 * g.stddev


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

def find_best_detector_position(photons, col='det_y', objective_func=sigma_clipped_std,
                                orientation=np.eye(3)):
    '''Numerically optimize the position of a detector to find the position of best focus/

    This routine places detectors at different positions and calculates the width of the
    photon distribution for each position.
    As a side effect, ``photons`` will be set to the intersection with the last try of
    the detector position.

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
        Rotation matrix for the detector. By default the detector is parallel to the yz plane
        of the global coordinate system (see `pos4d`).

    Returns
    -------
    opt : OptimizeResult
        see `scipy.optimize.minimize`
    '''
    def width(x, photons):
        mdet = FlatDetector(position=np.array([x, 0, 0]), orientation=orientation, zoom=1e5, pixsize=1.)
        photons = mdet.process_photons(photons)
        return objective_func(photons[col].data)

    return scipy.optimize.minimize(width, 0, args=(photons,), options={'maxiter': 20, 'disp': True})
