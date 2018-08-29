import inspect
from functools import wraps
import collections

import numpy as np
from transforms3d import affines, euler
from astropy.table import Table
from ..simulator import Parallel
from .uncertainties import generate_facet_uncertainty as genfacun
from ..analysis.gratings import resolvingpower_from_photonlist as resol
from ..analysis.gratings import AnalysisError


def oneormoreelements(func):
    '''Decorator for functions that modify optical elements.

    The functions in this module are written to work on a single optical
    element. This decorator allows them to accept a list of elements or
    a single element.
    '''
    @wraps(func)
    def func_wrapper(elements, *args, **kwargs):
        if isinstance(elements, collections.Iterable):
            for e in elements:
                func(e, *args, **kwargs)
        else:
            func(elements, *args, **kwargs)

    return func_wrapper


@oneormoreelements
def wiggle(e, dx=0, dy=0, dz=0, rx=0., ry=0., rz=0.):
    '''Move and rotate elements around principal axes.

    Parameters
    ----------
    e : `marxs.simulator.Parallel` or list of those elements
        Elements where uncertainties will be set
    dx, dy, dz : float
        accuracy of grating positioning in x, y, z (in mm) - Gaussian sigma, not FWHM!
    rx, ry, rz : float
        accuracy of grating positioning. Rotation around x, y, z (in rad) - Gaussian sigma, not FWHM!
    '''
    e.elem_uncertainty = genfacun(len(e.elements), [dx, dy, dz], [rx, ry, rz])
    e.generate_elements()


@oneormoreelements
def moveglobal(e, dx=0, dy=0, dz=0, rx=0., ry=0., rz=0.):
    '''Move and rotate origin of the whole `marxs.simulator.Parallel` object.

    Parameters
    ----------
    e :`marxs.simulator.Parallel` or list of those elements
        Elements where uncertainties will be set
    dx, dy, dz : float
        translation in x, y, z (in mm)
    rx, ry, rz : float
        Rotation around x, y, z (in rad)
    '''
    e.uncertainty = affines.compose([dx, dy, dz],
                                    euler.euler2mat(rx, ry, rz, 'sxyz'),
                                    np.ones(3))
    e.generate_elements()


@oneormoreelements
def moveindividual(e, dx=0, dy=0, dz=0, rx=0, ry=0, rz=0):
    '''Move and rotate all elements of `marxs.simulator.Parallel` object.

    Parameters
    ----------
    e :`marxs.simulator.Parallel` or list of those elements
        Elements where uncertainties will be set
    dx, dy, dz : float
        translation in x, y, z (in mm)
    rx, ry, rz : float
        Rotation around x, y, z (in rad)
    '''
    e.elem_uncertainty = [affines.compose((dx, dy, dz),
                                          euler.euler2mat(rx, ry, rz, 'sxyz'),
                                          np.ones(3))] * len(e.elements)
    e.generate_elements()


@oneormoreelements
def varyperiod(element, period_mean, period_sigma):
    '''Randomly draw different grating periods for different gratings

    This function needs to be called with gratings as parameters, e.g.

    >>> from marxs.optics import CATGrating
    >>> from marxs.design.tolerancing import varyperiod
    >>> grating = CATGrating(order_selector=None, d=0.001)
    >>> varyperiod(grating, 2e-4, 1e-5)

    and the parameters are expected to have two components:
    center and sigma of a Gaussian distribution for grating contstant d.

    Parameters
    ----------
    element :`marxs.optics.FlatGrating` or similar (or list of those elements)
        Elements where uncertainties will be set
    period_mean : float
        Center of Gaussian (in mm)
    period_sigma : float
        Sigma of Gaussian (in mm)
    '''
    if not hasattr(element, '_d'):
        raise ValueError(f'Object {element} does not have grating period `_d` attribute.')
    element._d = np.random.normal(period_mean, period_sigma)


@oneormoreelements
def varyorderselector(element, orderselector, *args, **kwargs):
    '''Modify the OrderSelector for a grating

    Parameters
    ----------
    element :`marxs.optics.FlatGrating` or similar (or list of those elements)
        Elements where the OrderSelector will be changed
    orderselector : class
        This should be a subclass of `InterpolateRalfTable` which determines how
        the order will be selected. In the case of the default class, the blaze
        angle of an incoming photons will be modified randomly to represent
        small-scale deviations from the flatness of the gratings.
    args, kwargs :
        All other parameters are used to initialize the OrderSelector
    '''
    if not hasattr(element, 'order_selector'):
        raise ValueError(f'Object {element} does not have an order_selector attribute.')
    element.order_selector = order_selector(*args, **kwargs)


@oneormoreelements
def varyscatter(element, inplanescatter, perplanescatter):
    '''Change the scatter properties of an SPO

    Parameters
    ----------
    elements :`marxs.optics.RadialMirrorScatter` or list of those elements
        Elements where uncertainties will be set
    inplanescatter : float
        new value for inplanescatter of mirror
    perpplanescatter : float
        new value for perpplanescater of mirror
    '''
    if not hasattr(element, 'inplanescatter'):
         raise ValueError(f'Object {element} does not have an inplanescatter attribute.')
    if not hasattr(element, 'perpplanescatter'):
         raise ValueError(f'Object {element} does not have an perpplanescatter attribute.')
    element.inplanescatter = inplanescatter
    element.perpplanescatter = perplanescatter


class CaptureResAeff(object):
    '''Capture resolving power and effective area for a tolerancing simulation.

    Instances of this class can be called with a list of input parameters for
    a tolerancing simulation and a resulting photon list. The photon list
    will be analysed for resolving power and effective area in a number of
    relevant orders.
    Every instance of this object has a ``tab`` attribute and every time the
    instance is called it adds one row of data to the table.

    Parameters
    ----------
    n_parameters : int
        Parameters will be stored in a vector-valued column called
        "Parameters". To generate this column in the correct size,
        ``n_parameters`` specifies the number of parameters.
    A_geom : number
        Geometric area of aperture for the simulations that this instance
        will analyze.
    on_detector_test : callable
        A function that can be called on a photon table and returns a
        boolean array with elements set to ``True`` for photons that hit
        the detector.
    '''
    orders = np.arange(-15, 5)

    order_col = 'order'
    '''Column names for grating orders'''

    dispersion_coord = 'proj_x'
    '''Dispersion coordinate for
    `marxs.analysis.gratings.resolvingpower_from_photonlist`'''

    def __init__(self, n_parameters, Ageom=1):
        self.Ageom = Ageom
        form = '{}f4'.format(len(self.orders))
        self.tab = Table(names=['Parameters', 'Aeff0', 'Aeffgrat', 'Rgrat', 'Aeff', 'R'],
                         dtype=['{}f4'.format(n_parameters),
                                float, float, float, form, form]
                         )
        for c in ['Aeff', 'Aeffgrat', 'Aeff']:
            self.tab[c].unit = Ageom.unit

    def find_photon_number(self, photons):
        '''Find the number of photons in the simulation.

        This method simply returns the length of the photons list which
        works if it has not been pre-filtered in any step.

        Subclasses can implement other ways, e.g. to inspect the header for a
        keyword.

        Parameters
        ----------
        photons : `astropy.table.Table`
            Photon list

        Returns
        -------
        n : int
            Number of photons
        '''
        return len(photons)

    def calc_result(self, filtered, n_photons):
        '''Calculate Aeff and R for an input photon list.

        Parameters
        ----------
        photons : `astropy.table.Table`
            Photon list. This photon list should already be filtered
            and contain only "good" photons, i.e. photons that hit a
            detector and have a non-zero probability.
        n_photons : number
            In order to calculate the effective area, the function needs
            to calculate the fraction of detected photons and to do that
            the total number of simulated photons needs to be passed in.

        Returns
        -------
        result : dict
            Dictionary with per-order Aeff and R, as well as values
            summed over all grating orders.
        '''
        aeff = np.zeros(len(self.orders))
        disporders = self.orders != 0

        if len(filtered) == 0:
            res = np.nan * np.ones(len(self.orders))
            avggratres = np.nan
        else:
            try:
                res, pos, std = resol(filtered, self.orders,
                                      col=self.dispersion_coord,
                                      zeropos=None, ordercol=self.order_col)
            except AnalysisError:
                # Something did not work, e.g. too few photons to find zeroth order
                res = np.nan * np.ones(len(self.orders))

            for i, o in enumerate(self.orders):
                aeff[i] = filtered['probability'][filtered[self.order_col] == o].sum()
            aeff = aeff / n_photons * self.Ageom
            # Division by 0 causes more nans, so filter those out
            # Also, res is nan if less than 20 photons are detected
            # so we need to filter those out, too.
            ind = disporders & (aeff > 0) & np.isfinite(res)
            if ind.sum() == 0:  # Dispersed spectrum misses detector
                avggratres = np.nan
            else:
                avggratres = np.average(res[ind],
                                        weights=aeff[ind] / aeff[ind].sum())
        # The following lines work for an empty photon list, too.
        aeffgrat = np.sum(aeff[disporders])
        aeff0 = np.sum(aeff[~disporders])
        return {'Aeff0': aeff0, 'Aeffgrat': aeffgrat, 'Aeff': aeff,
                'Rgrat': avggratres, 'R': res}

    def __call__(self, parameters, photons, n_photons):
        out = self.calc_result(photons, n_photons)
        out['Parameters'] = parameters
        self.tab.add_row(out)


def singletolerance(photons_in, instrum_before,
                    wigglefunc, wigglepars,
                    instrum_after, derive_result):
    photons = instrum_before(photons_in.copy())
    for i, pars in enumerate(wigglepars):
        print('Working on {}/{}'.format(i, len(wigglepars)))
        instrum = wigglefunc(pars)
        p_out = instrum(photons.copy())
        p_out = instrum_after(p_out)
        if 'det_x' in p_out.colnames:
            ind = np.isfinite(p_out['det_x']) & (p_out['probability'] > 0)
        else:
            ind = []
        derive_result(pars, p_out[ind], len(p_out))
