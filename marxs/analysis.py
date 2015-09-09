import numpy as np
import scipy.optimize
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats

from .optics import FlatDetector, constant_order_factory

def measure_FWHM(data):
    '''Obtain the FWHM of some quantity in an event list.

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
                                orientation=np.eye(3), **kwargs):
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
    kwargs : see `scipy.optimize.minimize`
        All other keyword argument will be passed to `scipy.optimize.minimize`.
    Returns
    -------
    opt : OptimizeResult
        see `scipy.optimize.minimize`
    '''
    def width(x, photons):
        mdet = FlatDetector(position=np.array([x, 0, 0]), orientation=orientation, zoom=1e5, pixsize=1.)
        photons = mdet.process_photons(photons)
        return objective_func(photons[col].data)

    return scipy.optimize.minimize(width, 0, args=(photons,), options={'maxiter': 20, 'disp': True},
                                   **kwargs)


def fwhm_per_order(gas, photons, orders=np.arange(-11,-1)):
    '''loop over grating orders and calculate FWHM in every order.

    For every order the detector position will be optimized numerically to give the smallest
    FWHM by moving it along the x-axis.
    As a side effect, the function that selects the grating orders for diffraction in ``gas``
    will be changed. Pass a deep copy of the GAS if this could affect consecutive computations.

    Parameters
    ----------
    gas : `marxs.design.rowland.GratingArrayStructure`
    photons : `astropy.table.Table`
        Photon list to be processed for every order
    orders : np.array of type int
        Order numbers

    Returns
    -------
    fwhm : np.array
        FWHM for each order
    det_x : np.array
        Detector x position which gives minimal FWHM.
    res : np.array
        Resolution defined as FWHM / position for each order.
    '''
    res = np.zeros_like(orders, dtype=float)
    fwhm = np.zeros_like(orders, dtype=float)
    det_x = np.zeros_like(orders, dtype=float)

    for i, order in enumerate(orders):
        gratingeff = constant_order_factory(order)
        gas.elem_args['order_selector'] = gratingeff
        for facet in gas.elements:
            facet.order_selector = gratingeff

        pg = photons.copy()
        pg = gas.process_photons(pg)
        pg = pg[pg['order'] == order]  # Remove photons that slip between the gratings
        xbest = find_best_detector_position(pg, objective_func=measure_FWHM)
        fwhm[i] = xbest.fun
        det_x[i] = xbest.x
        meanpos, medianpos, stdpos = sigma_clipped_stats(pg['det_y'])
        res[i] = np.abs(meanpos / xbest.fun)
    return fwhm, det_x, res


def weighted_per_order(data, orders, energy, gratingeff):
    '''Summarize a per-order table of a quantity such as spectral resolution.

    `marxs.analysis.fwhm_per_order` produces a set of data for each grating order,
    most notably the spectral resolution achieved in every spectral order for every energy.
    In practice, however, most orders see only a very small number of photons and will not
    contribute significantly to the observed signal.

    This method provides one way to summarize the data by calculating the weighted mean
    of the resolution fir each energy, weighted by the probability of photons for be
    diffracted into that order ( = the expected fraction).

    Parameters
    ----------
    data : (N, M) np.array
        Array with data for N orders and M energies, e.g. spectral resolution
    orders : np.array of length N of type int
        Order numbers
    energy : np.array for length M of type float
        Energies for each entry in resolution
    gratingeff : `marxs.optics.gratings.EfficiencyFile`
        Object that holds the probability that a photon on a certain energy is
        diffracted to a specific order.

    Returns
    -------
    weighted_res : np.array of length M
        Weighted resolution for each energy
    '''
    if len(orders) != data.shape[0]:
        raise ValueError('First dimension of "data" must match length of "orders".')
    if len(energy) != data.shape[1]:
        raise ValueError('Second dimension of "data" must match length of "energy".')

    weights = np.zeros_like(data)
    for i, o in enumerate(orders):
        ind_o = (gratingeff.orders == o).nonzero()[0]
        if len(ind_o) != 1:
            raise KeyError('No data for order {0} in gratingeff'.format(o))
        en_sort = np.argsort(gratingeff.energy)
        weights[o, :] = np.interp(energy, gratingeff.energy[en_sort],
                                  gratingeff.prob[:, ind_o[0]][en_sort])

    return np.ma.average(data, axis=0, weights=weights)
