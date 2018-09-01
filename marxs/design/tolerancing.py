import inspect
from functools import wraps
import collections

import numpy as np
from transforms3d import affines, euler
from astropy.table import Table
from ..simulator import Parallel
from .uncertainties import generate_facet_uncertainty as genfacun
from ..analysis.gratings import (resolvingpower_from_photonlist,
                                 effectivearea_from_photonlist)
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
def varyattribute(element, **kwargs):
    '''Modify the attributes of an element.

    This function modifies the attributes of an object. The keyword arguments
    are name and value of the attributes to be changed. This can be used for
    a wide variety of MARXS objects where the value of a parameter is stored
    as an instance attribute.

    Parameters
    ----------
    element : object
        Some optical component.
    keywords : it depends

    Examples
    --------
    In a `marxs.optics.RadialMirrorScatter` object, the width of the scattering
    can be modified like this:

    >>> from marxs.optics import RadialMirrorScatter
    >>> from marxs.design.tolerancing import varyattribute
    >>> rms = RadialMirrorScatter(inplanescatter=1e-5, perpplanescatter=0)
    >>> varyattribute(rms, inplanescatter=2e-5)

    It is usually easy to see which attributes of a
    `marxs.simulator.SimulationSequenceElement` are relevant to be changed in
    tolerancing simulations when looking at the attributes or the implementation
    of those elements.
    '''
    for key, val in kwargs.items():
        if not hasattr(element, key):
            raise ValueError(f'Object {element} does not have {key} attribute.')
        setattr(element, key, val)


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
def varyorderselector(element, order_selector, *args, **kwargs):
    '''Modify the OrderSelector for a grating

    Parameters
    ----------
    element :`marxs.optics.FlatGrating` or similar (or list of those elements)
        Elements where the OrderSelector will be changed
    order_selector : class
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


def run_tolerances(photons_in, instrum, wigglefunc, wiggleparts,
                   parameters, analyzefunc):
    '''Run tolerancing calculations for a range of parameters

    This function takes an instrument configuration and a function to change
    one aspect of it. For every change it runs a simulations and calculates a
    figure of merit. As the name indicates, this function is designed to derive
    alignment tolerances for certain instrument parts but it might be general
    enough for other parameter studies in instrument design.

    Parameters
    ----------
    photons_in : `astropy.table.Table`
        Input photons list. To speed up the computation, it is useful if photon
        list has been run through all elements of the instrument that are
        located before the first element that is toleranced here.
    instrum : `marxs.simulator.Sequence` object
        An instance of the instrument which contains all elements that
        `photons_in` still have to pass. This can include elements that are
        toleranced in this run and those that are located behind them.
    wigglefunc : callable
        Function that modifies the `instrum` with the following calling signature:
        ``wigglefunc(wiggleparts, pars)`` where ``pars`` is one dict from
        `parameters`. Note that this function is called with `wiggleparts` which
        can be the same as `instrum` or just a subset.
    wiggleparts :  `marxs.base.SimulationSequenceElement` instance
        Element which is modified by `wigglefunc`. Typically this is a subset of
        `instrum`. For example, to tolerance the mirror alignments
        `wiggleparts` would just be the mirror objects, while `instrum` would
        contain all the parts of the instrument that the photons need to run
        though up to the detector.
    parameters : list of dicts
        List of parameter values for calls to `wigglefunc`.
    analyzefunc : callable function or object
        This is called


    Returns
    -------
    result : list of dicts
        Each dict contains parameters and results for one run.

    Notes
    -----
    The format of input and output as lists of dicts is chosen because this
    would work well for a parallel version of this function which could have
    the same interface when it is implemented.
    '''
    out = []
    for i, pars in enumerate(parameters):
        print(f'Working on simulation {i}/{len(parameters)}')
        wigglefunc(wiggleparts, **pars)
        photons = instrum(photons_in.copy())
        pars.update(analyzefunc(photons))
        out.append(pars)

    return out


class CaptureResAeff():
    '''Capture resolving power and effective area for a tolerancing simulation.

    Instances of this class can be called with a photon list for a tolerancing
    simulation. The photon list will be analysed for resolving power and
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
        `marxs.analysis.gratings.resolvingpower_from_photonlist`

    '''
    def __init__(self, A_geom=1, order_col='order',
                 orders=np.arange(-10, 11), dispersion_coord='det_x'):
        self.A_geom = A_geom
        self.order_col = order_col
        self.orders = np.asanyarray(orders)
        self.dispersion_coord = dispersion_coord

    def __call__(self, photons):
        '''Calculate Aeff and R for an input photon list.

        Parameters
        ----------
        photons : `astropy.table.Table`
            Photon list.

        Returns
        -------
        result : dict
            Dictionary with per-order Aeff and R, as well as values
            summed over all grating orders.
        '''
        aeff = effectivearea_from_photonlist(photons, self.orders, len(photons),
                                             self.A_geom, self.order_col)
        try:
            ind = (np.isfinite(photons[self.dispersion_coord]) &
                   (photons['probability'] > 0))
            res, pos, std = resolvingpower_from_photonlist(photons, self.orders,
                                                           col=self.dispersion_coord,
                                                           zeropos=None,
                                                           ordercol=self.order_col)
        except AnalysisError:
            # Something did not work, e.g. too few photons to find zeroth order
            res = np.nan * np.ones(len(self.orders))

        disporders = self.orders != 0
        # The following lines work for an empty photon list, too.
        aeffgrat = np.sum(aeff[disporders])
        aeff0 = np.sum(aeff[~disporders])

        # Division by 0 causes more nans, so filter those out
        # Also, res is nan if less than 20 photons are detected
        # so we need to filter those out, too.
        ind = disporders & (aeff > 0) & np.isfinite(res)
        if ind.sum() == 0:  # Dispersed spectrum misses detector
            avggratres = np.nan
        else:
            avggratres = np.average(res[ind],
                                    weights=aeff[ind] / aeff[ind].sum())
        return {'Aeff0': aeff0, 'Aeffgrat': aeffgrat, 'Aeff': aeff,
                'Rgrat': avggratres, 'R': res}
