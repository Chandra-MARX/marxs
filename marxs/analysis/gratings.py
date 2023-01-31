# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from astropy.stats import sigma_clipped_stats
import astropy.units as u

from ..optics import FlatDetector, CircularDetector, OrderSelector
from ..design import RowlandTorus
from . import (find_best_detector_position)
from .analysis import sigma_clipped_std
from ..math.geometry import Cylinder

__all__ = ['AnalysisError',
           'resolvingpower_per_order',
           'resolvingpower_from_photonlist',
           'resolvingpower_from_photonlist_robust',
           'weighted_per_order',
           'average_R_Aeff',
           'CaptureResAeff',
           'CaptureResAeff_CCDgaps',
           ]

class AnalysisError(Exception):
    pass


def resolvingpower_per_order(gratings, photons, orders, detector=None,
                             colname='det_x'):
    '''Calculate the resolving power in every grating order.

    As input this function takes a Grating Array Structure (``gratings``) and
    a list of photons ready to hit the grating, i.e. the photons have already
    passed through aperture and mirror in e.g. a Chandra-like design. The
    function will take the same input photons and send them through the gas
    looping over the grating orders. In each step of the loop all photons are
    send to the same order, thus the statistical uncertainty on the measured
    spectral resolving power (calculated as ``pos_x / FWHM_x``) is the same
    for every order and is set by the number of photons in the input list.

    As a side effect, the function that selects the grating orders for
    diffraction in ``gas`` will be changed. Pass a deep copy of the GAS if
    this could affect consecutive computations.

    Unlike `~marxs.analysis.gratings.resolvingpower_from_photonlist` this
    function ray-traces the input photon list from the gratings to the
    detectors.

    Parameters
    ----------
    gratings : `marxs.simulator.Parallel`
        Structure that holds individual grating elements,
        e.g. `marxs.design.GratingArrayStructure`
    photons : `astropy.table.Table`
        Photon list to be processed for every order
    orders : np.array of type int
        Order numbers
    detector : marxs optical element or `marxs.design.RowlandTorus` or
        ``None``. Photons are projected onto a detector. There are three ways
        to define this detector:

        - Pass in an instance of an optical element (e.g. a
          `marxs.optics.FlatDetector`).
        - Pass in a `marxs.design.RowlandTorus`. This function will generate
          a detector that follows the Rowland circle.
        - ``None``. A flat detector in the yz plane is used, but the x position
          for this detector is numerically optimized in each step.

    colname : string
        Name of the column that labels the dispersion direction. This is only
        used if a detector instance is passed in explicitly.

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
        col = 'det_x'  # 0 at center, detpix_x is 0 in corner.
        zeropos = 0.
    elif isinstance(detector, RowlandTorus):
        det = CircularDetector(geometry=Cylinder.from_rowland(detector,
                                                              width=1e5,
                                                              rotation=np.pi,
                                                              kwargs={'phi_lim':[-np.pi/2, np.pi/2]}))
        info['method'] = 'Circular detector on Rowland circle'
        col = 'detpix_x'
        # Find position of order 0.
        # if Rowland torus is tilted, might not be at phi=0.
        pg = photons.copy()
        pg = det(pg)
        # Remove photons that miss a mirror etc.
        pg = pg[pg['probability'] > 0.]
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
        gratingeff = OrderSelector([order])
        gratings.elem_args['order_selector'] = gratingeff
        for elem in gratings.elements:
            elem.order_selector = gratingeff

        pg = photons.copy()
        pg = gratings(pg)
        # Remove photons that slip between the gratings
        pg = pg[(pg['order'] == order) & (pg['probability'] > 0.)]
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

    `marxs.analysis.fwhm_per_order` produces a set of data for each grating
    order, most notably the spectral resolution achieved in every spectral
    order for every energy. In practice, however, most orders see only a very
    small number of photons and will not contribute significantly to the
    observed signal.

    This method provides one way to summarize the data by calculating the
    weighted mean of the resolution for each energy, weighted by the
    probability of photons to be diffracted into that order ( = the expected
    fraction).

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


def average_R_Aeff(r, aeff, axis=None):
    '''Average R and Aeff over one dimension e.g. channel or orders

    Aeff is summed over the input, while R is calculated as the
    weighted average, using Aeff as the weight for each order.
    This function works for masked arrays and those with nan's.

    Unlike `weighted_per_order` this function summarizes measured
    quantities, so instead of an expected fraction that will be interpolated
    from some table, this uses the observed effective area as weight.

    Parameters
    ----------
    r : array
        Resolving power
    aeff : array
        Effective area
    axis : `None` or int
        Passed to `np.ma.average`. This can be used if the input arrays
        are multi-dimensional.

    Returns
    -------
    aeff_avg : array or float
        Summed effective area
    res_avg : array or float
        Average resolving power
    '''
    aeff_avg = aeff.sum(axis=axis)
    res_avg = np.ma.average(np.ma.masked_invalid(r),
                            weights=aeff, axis=axis)
    return res_avg, aeff_avg


def resolvingpower_from_photonlist(photons, orders,
                                   col='proj_x', zeropos=None,
                                   ordercol='order'):
    '''Calculate the resolving power for several grating orders

    If fewer than 20 photons are detected in a single order, this function
    returns nan values.

    Unlike `~marxs.analysis.gratings.resolvingpower_per_order` this method does
    not run any ray-trace simulations. It just extracts information from a
    photon list that is passed in.

    Parameters
    ----------
    photons : `astropy.table.Table`
        Photon event list
    orders : np.array
        Orders for which the resolving power will be calculated
    col : string
        Column name for the column holding the dispersion coordinate.
    zeropos : float or ``None``
        Value of column `col` where the zeroth order is found. If not given,
        this is calculated (assuming the zeroth order photons are part of the
        event list).
    ordercol : string
        Name of column that lists grating order for each photon

    Returns
    -------
    res : np.array
        resolving power for each order
    pos : np.array
        mean value of ``col`` for each order
    std : np.array
        standard deviation of the distribution of ``col`` for each order
    '''
    if zeropos is None:
        ind = (photons[ordercol] == 0)
        if ind.sum() < 20:
            raise AnalysisError('Too few photons in list to determine position of order 0 automatically.')
        zeropos, medzeropos, stdzero = sigma_clipped_stats(photons[col][ind])

    pos = np.zeros_like(orders, dtype=float)
    std = np.zeros_like(orders, dtype=float)

    for i, o in enumerate(orders):
        ind = (photons[ordercol] == o)
        if ind.sum() > 20:
            # There is a .value here to work around https://github.com/astropy/astropy/issues/13281
            meanpos, medianpos, stdpos = sigma_clipped_stats(photons[col][ind].value)
        else:
            meanpos, stdpos = np.nan, np.nan
        pos[i] = meanpos
        std[i] = stdpos
    res = np.abs(pos - zeropos) / (std * 2.3548)
    return res, pos, std


def resolvingpower_from_photonlist_robust(lphotons, orders,
                                          cols, zeropositions,
                                          ordercol='order'):
    '''Robustly calculate the resolving power for grating orders

    This function wraps `resolvingpower_from_photonlist` to iterate
    over several columns. It calculates the resolving power R for several
    columns and reports the worst case.

    We often want to determine R from the distribution of photons on a CCD.
    Geometry effects like CCDs that are slightly off the
    Rowland torus make R calculated on the detectors lower than an ideal case.
    However, that R can spike if part of the line hits a
    chip gap. So, we can calculate R for a continuous detector on the
    ideal Rowland circle and report that R in such a case.

    Parameters
    ----------
    lphotons : list of `astropy.table.Table`
        List of Photon event list
    orders : np.array
        Orders for which the resolving power will be calculated
    cols : list
        List of column names for the column holding the dispersion coordinate.
    zeropositions : list
        List of values of column `col` where the zeroth order is found. If not given,
        this is calculated (assuming the zeroth order photons are part of the
        event list).
    ordercol : string
        Name of column that lists grating order for each photon

    Returns
    -------
    res : np.array
        resolving power for each order
    pos : np.array
        mean value of ``col`` for each order
    std : np.array
        standard deviation of the distribution of ``col`` for each order
    '''
    if len(cols) != len(zeropositions):
        raise ValueError('Number of elements in cols and zeropositions is not the same.')
    if len(cols) != len(lphotons):
        raise ValueError('Number of elements in cols and photon lists is not the same.')

    results = np.zeros((len(cols), len(orders)))
    positions = np.zeros_like(results)
    stds = np.zeros_like(results)
    for i, (photons, col, zeropos) in enumerate(zip(lphotons, cols, zeropositions)):
        res, pos, std = resolvingpower_from_photonlist(photons, orders,
                                                    col=col, zeropos=zeropos,
                                                    ordercol=ordercol)
        results[i, :] = res
        positions[i, :] = pos
        stds[i, :] = std
    ind = np.argmin(results, axis=0)
    ind_n = np.arange(results.shape[1])
    return results[ind, ind_n], positions[ind, ind_n], stds[ind, ind_n]


def effectivearea_from_photonlist(photons, orders, n_photons, A_geom=1. * u.cm**2,
                                  ordercol='order'):
    '''Calculate the effective area several grating orders

    This is based on the probabilities of the photons in the list, so
    this photons list must already account for all instrument
    components, for example, do not forget to set the probability to 0
    for photons that fall into a chip gap.

    Parameters
    ----------
    photons : `astropy.table.Table`
        Photon event list
    orders : np.array
        Orders for which the resolving power will be calculated
    n_photons : int
        Number of photons originally simulated
    A_geom : quantity
        Geometric area of aperture that was used for photon list.
    ordercol : string
        Name of column that lists grating order for each photon

    Returns
    -------
    aeff : np.array
        effective area for each order
    '''
    aeff = np.zeros(len(orders))
    for i, o in enumerate(orders):
        aeff[i] = photons['probability'][photons[ordercol] == o].sum()
    return aeff / n_photons * A_geom


def identify_photon_in_subaperture(angle, max_ang, ang_0=np.pi / 2):
    '''Find which photons are in a sub-aperture of two mirrored sectors

    This function includes mirroring on the 0 line, e.g. if
    max_ang=20 deg and ang_0 = 90 deg, includes photons in two sectors
    from 70-110 deg and 250-290 deg.

    Parameters
    ----------
    angle : array
        Angles (in rad) to be tested
    max_ang : float
        Maximal distance (in rad) from ``ang_0`` that is included in the sub-aperturing
    ang_0 : float
        Center of the sub-aperturing region. Default is sub-aperturing perpendicular
        to the ``angle=0`` line, which is the most common sub-aperturing layout with
        subaperturing perpendicular to the dispersion direction as ``angle=0``.
    '''
    indplus = np.isclose(np.mod(angle, 2 * np.pi), ang_0, atol=max_ang)
    indminus = np.isclose(np.mod(angle, 2 * np.pi), 2 * np.pi - ang_0, atol=max_ang)
    return indplus | indminus


class CaptureResAeff():
    '''Capture resolving power and effective area for a simulation.

    Instances of this class can be called with a photon list for a
    simulation. The photon list will be analyzed for resolving power and
    effective area in a number of relevant orders.

    This is implemented as a class and not a simple function. When the class is
    initialized a number of parameters that are true for any to the
    analysis (e.g. the names of certain columns) are set and saved in the class
    instance.

    The implementation of this class is geared towards instruments with
    gratings but can also serve as an example how a complex analysis that
    derives several different parameters can be implemented.

    Results for effective area and resolving power are reported on a per order
    basis and also summarized for all grating orders combined and the zeroth
    order separately.

    Inherit form this class to customize the filters applied to the photon list
    before effective area and resolving power are calculated.

    Parameters
    ----------
    A_geom : number
        Geometric area of aperture for the simulations that this instance
        will analyze.
    order_col : string
        Column names for grating orders
    orders : array
        Order numbers to consider in the analysis
    dispersion_coord : string
        Dispersion coordinate for
        `marxs.analysis.gratings.resolvingpower_from_photonlist`. Any photons
        that have non-finite values or are masked in this column will be ignored
        for the purpose of the resolving power calculation.
    zeropos : float
        Position of zeroth order in ``dispersion_coord``
    '''
    def __init__(self, A_geom=1, order_col='order',
                 orders=np.arange(-10, 11),
                 dispersion_coord='det_x',
                 zeropos=None):
        self.A_geom = A_geom
        self.order_col = order_col
        self.orders = np.asanyarray(orders)
        self.dispersion_coord = dispersion_coord
        self.zeropos = zeropos

    def aeff_filter(self, photons):
        '''Filter photon list before calculating the effective area.

        Here: No-op (all photons are used.)

        Parameters
        ----------
        photons : `astropy.table.Table`
            Photon list.

        Returns
        -------
        ind : numpy array
           array of boolean values that can be used to index ``photons`` and select
           a sub-set to be used to calculate the effective area.
        '''
        return np.ones(len(photons), dtype=bool)

    def res_filter(self, photons):
        ind = (np.isfinite(photons[self.dispersion_coord]) &
                (photons['probability'] > 0))
        if hasattr(photons[self.dispersion_coord], 'mask'):
            ind = ind & ~photons[self.dispersion_coord].mask
        return ind

    def __call__(self, photons, n_photons=None):
        '''Calculate Aeff and R for an input photon list.

        Parameters
        ----------
        photons : `astropy.table.Table`
            Photon list.
        n_photons : int or `None`
            Number of photons originally simulated. If ``None`` use length of
            ``photons``.

        Returns
        -------
        result : dict
            Dictionary with per-order Aeff and R, as well as values
            summed over all grating orders.
        '''
        if n_photons is None:
            n_photons = len(photons)
        ind = self.aeff_filter(photons)
        aeff = effectivearea_from_photonlist(photons[ind], self.orders, n_photons,
                                             self.A_geom, self.order_col)
        try:
            ind = self.res_filter(photons)
            res, pos, std = resolvingpower_from_photonlist(photons[ind], self.orders,
                                                           col=self.dispersion_coord,
                                                           zeropos=self.zeropos,
                                                           ordercol=self.order_col)
        except AnalysisError:
            # Something did not work, e.g. too few photons to find zeroth order
            res = np.nan * np.ones(len(self.orders))

        disporders = self.orders != 0
        avggratres, aeffgrat = average_R_Aeff(res[disporders],
                                              aeff[disporders])
        aeff0 = np.sum(aeff[~disporders])
        return {'Aeff0': aeff0, 'Aeffgrat': aeffgrat, 'Aeff': aeff,
                'Rgrat': avggratres, 'R': res}


class CaptureResAeff_CCDgaps(CaptureResAeff):
    '''Capture resolving power and effective area for a tolerancing simulation.

    Unlike `CaptureResAeff` this objects is set up to take into account CCD gaps.
    For the resolving power calculation, one wants to ignore CCD gaps, because they
    might make the observed LSF artificially small - if only half of the LSF falls
    on a CCD, and the other photons are not counted, the LSF will appear only
    half as wide. In contrast, for the effective area calculation, we *do* want to
    ignore photons that miss a CCD.

    Here, we following approach is taken: This class overrides
    `~marxs.design.tolerancing.CaptureResAeff.aeff_filter`.
    All photons with a probability >0 are presumed ot reach the focal plane.
    For the effective area, an additional
    filter is applied: Only photons with `photons['aeff_filter_col'] > 0` will
    be used. Typically, this could be the CCD_ID.

    Parameters
    ----------
    aeff_filter_col : string
        Only photons where the value in this column is larger than 0 will be used
        for the calculation of the effective area.

    kwargs :
        All other keyword arguments are passed to `CaptureResAeff`.
    '''
    def __init__(self, aeff_filter_col='CCD_ID', **kwargs):
        super().__init__(**kwargs)
        self.aeff_filter_col = aeff_filter_col

    def aeff_filter(self, photons):
        '''Filter photon list before calculating the effective area.

        Only photons with ``photons[aeff_filter_col] >= 0`` are used.

        Parameters
        ----------
        photons : `astropy.table.Table`
            Photon list.

        Returns
        -------
        ind : numpy array
           array of boolean values that can be used to index ``photons`` and select
           a sub-set to be used to calculate the effective area.
        '''
        return photons[self.aeff_filter_col] >= 0