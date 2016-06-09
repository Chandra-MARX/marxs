from __future__ import division

import numpy as np
import scipy.optimize
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats

from .optics import FlatDetector, CircularDetector, constant_order_factory

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
    hist, bin_edges = np.histogram(data, range=mean + np.array([-3, 3]) * std, bins = int(n/10))
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


def find_best_detector_position(photons, col='det_x', objective_func=sigma_clipped_std,
                                orientation=np.eye(3), **kwargs):
    '''Numerically optimize the position of a detector to find the position of best focus.

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


def resolvingpower_per_order(gratings, photons, orders=np.arange(-11,-1), rowland=None):
    '''Loop over grating orders and calculate the resolving power in every order.

    As input this function takes a Grating Array Structure (``gratings``) and a list of photons
    ready to hit the grating, i.e. the photons have already passed through aperture and
    mirror in e.g. a Chandra-like design. The function will take the same input photons and
    send them through the gas looping over the grating orders. In each step of the loop all
    photons are send to the same order, thus the statistical uncertainty on the measured
    spectral resolving power (calculated as ``pos_x / FWHM_x``)
    is the same for every order and is set by the number of photons in the input list.

    As a side effect, the function that selects the grating orders for diffraction in ``gas``
    will be changed. Pass a deep copy of the GAS if this could affect consecutive computations.

    Parameters
    ----------
    gratings : `marxs.simulator.Parallel`
        Structure that holds individual grating elements,
        e.g. `marxs.design.GratingArrayStructure`
    photons : `astropy.table.Table`
        Photon list to be processed for every order
    orders : np.array of type int
        Order numbers
    rowland : `marxs.design.RowlandTorus` or ``None``.
        Photons are projected onto a detector. If ``rowland`` is given, the detector
        will be placed on the Rowland torus appropriately; if it is ``None``
        the best detector position is determined numerically assuming that the
        detector should be parallel to the yz plane.

    Returns
    -------
    res : np.array
        Resolution defined as FWHM / position for each order.
    fwhm : np.array
        FWHM for each order
    info : dict
        Dictionary with more information, e.g. fit results
    '''
    res = np.zeros_like(orders, dtype=float)
    fwhm = np.zeros_like(orders, dtype=float)
    info = {}

    if rowland is not None:
        det = CircularDetector.from_rowland(rowland, width=1e5)
        info['method'] = 'Circular detector on Rowland circle'
        col = 'detpix_x' # 0 at center
    else:
        info['method'] = 'Detector position numerically optimized'
        info['fit_results'] = []
        col = 'det_x' # 0 at center, detpix_x is 0 in corner.

    for i, order in enumerate(orders):
        gratingeff = constant_order_factory(order)
        gratings.elem_args['order_selector'] = gratingeff
        for elem in gratings.elements:
            elem.order_selector = gratingeff

        pg = photons.copy()
        pg = gratings.process_photons(pg)
        pg = pg[pg['order'] == order]  # Remove photons that slip between the gratings
        if rowland is None:
            xbest = find_best_detector_position(pg, objective_func=measure_FWHM)
            info['fit_results'].append(xbest)
            det = FlatDetector(position=np.array([xbest.x, 0, 0]), zoom=1e5)
        pg = det(pg)
        meanpos, medianpos, stdpos = sigma_clipped_stats(pg[col])
        fwhm[i] = 2.3548 * stdpos
        res[i] = np.abs(meanpos / fwhm[i])
    return res, fwhm, info


def weighted_per_order(data, orders, energy, gratingeff):
    '''Summarize a per-order table of a quantity such as spectral resolution.

    `marxs.analysis.fwhm_per_order` produces a set of data for each grating order,
    most notably the spectral resolution achieved in every spectral order for every energy.
    In practice, however, most orders see only a very small number of photons and will not
    contribute significantly to the observed signal.

    This method provides one way to summarize the data by calculating the weighted mean
    of the resolution for each energy, weighted by the probability of photons for be
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
