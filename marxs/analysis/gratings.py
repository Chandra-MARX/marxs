from __future__ import division

import numpy as np
from astropy.stats import sigma_clipped_stats

from ..optics import FlatDetector, CircularDetector, constant_order_factory
from ..design import RowlandTorus
from . import (find_best_detector_position)
from .analysis import sigma_clipped_std


class AnalysisError(Exception):
    pass

def resolvingpower_per_order(gratings, photons, orders, detector=None, colname='det_x'):
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

    Unlike `~marxs.analysis.gratings.resolvingpower_from_photonlist` this function ray-traces
    the input photon list from the gratings to the detectors.

    Parameters
    ----------
    gratings : `marxs.simulator.Parallel`
        Structure that holds individual grating elements,
        e.g. `marxs.design.GratingArrayStructure`
    photons : `astropy.table.Table`
        Photon list to be processed for every order
    orders : np.array of type int
        Order numbers
    detector : marxs optical element or `marxs.design.RowlandTorus` or ``None``.
        Photons are projected onto a detector. There are three ways to define this detector:

        - Pass in an instance of an optical element (e.g. a `marxs.optics.FlatDetector`).
        - Pass in a `marxs.design.RowlandTorus`. This function will generate a detector that
          follows the Rowland circle.
        - ``None``. A flat detector in the yz plane is used, but the x position for this
          detector is numerically optimized in each step.

    colname : string
        Name of the column the labels the dispersion direction. This is only used if a detector
        instance is passed in explicitly.

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

    if detector is None:
        info['method'] = 'Detector position numerically optimized'
        info['fit_results'] = []
        col = 'det_x' # 0 at center, detpix_x is 0 in corner.
        zeropos = 0.
    elif isinstance(detector, RowlandTorus):
        det = CircularDetector.from_rowland(detector, width=1e5)
        info['method'] = 'Circular detector on Rowland circle'
        col = 'detpix_x'
        # Find position of order 0.
        # if Rowland torus is tilted, might not be at phi=0.
        pg = photons.copy()
        pg = det(pg)
        zeropos, temp1, temp2 = sigma_clipped_stats(pg[col])
    else:
        det = detector
        info['method'] = 'User defined detector'
        col = colname
        # Find position of order 0.
        pg = photons.copy()
        pg = det(pg)
        zeropos, temp1, temp2 = sigma_clipped_stats(pg[col])

    for i, order in enumerate(orders):
        gratingeff = constant_order_factory(order)
        gratings.elem_args['order_selector'] = gratingeff
        for elem in gratings.elements:
            elem.order_selector = gratingeff

        pg = photons.copy()
        pg = gratings.process_photons(pg)
        pg = pg[pg['order'] == order]  # Remove photons that slip between the gratings
        if detector is None:
            xbest = find_best_detector_position(pg, objective_func=sigma_clipped_std)
            info['fit_results'].append(xbest)
            det = FlatDetector(position=np.array([xbest.x, 0, 0]), zoom=1e5)
        pg = det(pg)
        meanpos, medianpos, stdpos = sigma_clipped_stats(pg[col])
        fwhm[i] = 2.3548 * stdpos
        res[i] = np.abs((meanpos - zeropos) / fwhm[i])
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
        weights[i, :] = np.interp(energy, gratingeff.energy[en_sort],
                                  gratingeff.prob[:, ind_o[0]][en_sort])

    return np.ma.average(data, axis=0, weights=weights)


def resolvingpower_from_photonlist(photons, orders, filterfunc=None, col='proj_x', zeropos=None,
                    ordercol='order'):
    '''Calculate the resolving power for several grating orders

    If fewer than 20 photons are detected in a single order, this function returns
    nan values.

    Unlike `~marxs.analysis.gratings.resolvingpower_per_order` this method does not run
    any ray-trace simulations. It just extract information from a photon list that is
    passed in.

    Parameters
    ----------
    photons : `astropy.table.Table`
        Photon event list
    orders : np.array
        Orders for which the resolving power will be calculated
    filterfunc : callable or ``None``
        If not ``None``, a function that takes the photon table and returns an
        index array. This can be used, e.g. filter out photons that hit
        particular CCDs or hot columns.
    col : string
        Column name for the column holding the dispersion coordinate.
    zeropos : float or ``None``
        Value of column `col` where the zeroth order is found. If not given, this is
        calculated (assuming the zeroth order photons are part of the event list).
    ordercol : string
        Name of column that lists grating order for each photon

    Returns
    -------
    res : np.array
        resolving power for each order
    pos : np.array
        mean value or ``col`` for each order
    std : np.array
        standard deviation of the distribution of ``col`` for each order
    '''
    if zeropos is None:
        ind = (photons[ordercol] == 0)
        if filterfunc is not None:
            ind = ind & filterfunc(photons)
        if ind.sum() < 20:
            raise AnalysisError('Too few photons in list to determine position of order 0 automatically.')
        zeropos, medzeropos, stdzero = sigma_clipped_stats(photons[col][ind])

    pos = np.zeros_like(orders, dtype=float)
    std = np.zeros_like(orders, dtype=float)

    for i, o in enumerate(orders):
        ind = (photons[ordercol] == o)
        if filterfunc is not None:
            ind = ind & filterfunc(photons)

        if ind.sum() > 20:
            meanpos, medianpos, stdpos = sigma_clipped_stats(photons[col][ind])
        else:
            meanpos, stdpos = np.nan, np.nan
        pos[i] = meanpos
        std[i] = stdpos
    res = np.abs(pos - zeropos) / (std * 2.3548)
    return res, pos, std
