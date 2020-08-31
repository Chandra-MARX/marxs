# Licensed under GPL version 3 - see LICENSE.rst
import inspect
from functools import wraps
import collections
from copy import copy
import warnings

import numpy as np
from transforms3d import affines, euler
from astropy.table import Table
from astropy import table
import astropy.units as u
from ..simulator import Parallel
from .uncertainties import generate_facet_uncertainty as genfacun
from ..analysis.gratings import (resolvingpower_from_photonlist,
                                 effectivearea_from_photonlist)
from ..analysis.gratings import AnalysisError

__all__ = ['oneormoreelements',
           'wiggle', 'moveglobal', 'moveindividual',
           'varyperiod', 'varyorderselector', 'varyattribute',
           'run_tolerances', 'run_tolerances_for_energies',
           'CaptureResAeff',
           'generate_6d_wigglelist',
           'select_1dof_changed',
           'plot_wiggle', 'load_and_plot',
           ]

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

    Unlike `~marxs.design.tolerancing.moveglobal` this does not move to
    center of the `~marxs.simulator.Parallel` object, instead it moves
    all elements individually. Unlike `~marxs.design.tolerancing.wiggle`,
    each element is moved by the same amount, not a number drawn from from
    distribution.

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

    >>> from astropy import units as u
    >>> from marxs.optics import RadialMirrorScatter
    >>> from marxs.design.tolerancing import varyattribute
    >>> rms = RadialMirrorScatter(inplanescatter=2 * u.arcsec, perpplanescatter=0 * u.arcsec)
    >>> varyattribute(rms, inplanescatter=1 * u.arcsec)

    It is usually easy to see which attributes of a
    `marxs.simulator.SimulationSequenceElement` are relevant to be changed in
    tolerancing simulations when looking at the attributes or the implementation
    of those elements.
    '''
    for key, val in kwargs.items():
        # check is needed because key could be misspelled so that we set an attribute
        # that did not exist before and that is never used
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
        This is called with a photon table and should return a dictionary
        of results. This could be, e.g. a
        `marxs.design.tolerancing.CaptureResAeff` instance.


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
        cpars = copy(pars)
        cpars.update(analyzefunc(photons))
        out.append(cpars)

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
            res, pos, std = resolvingpower_from_photonlist(photons[ind], self.orders,
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


def run_tolerances_for_energies(source, energies,
                                instrum_before, instrum_remaining,
                                wigglefunc, wiggleparts,
                                parameters, analyzefunc,
                                reset=None, t_source=1):
    '''Run tolerancing for a grid of parameters and energies

    This function loops over `~marxs.design.tolerancing.run_tolerances` for
    different energies.
    This function takes an instrument configuration and a function to change
    one aspect of it. For every change it runs a simulations and calculates a
    figure of merit. As the name indicates, this function is designed to derive
    alignment tolerances for certain instrument parts but it might be general
    enough for other parameter studies in instrument design.

    There are two nested loops here looping over energy and tolerancing
    parameters. In this case, the outer loop is over energies and then for
    every energy, `~marxs.design.tolerancing.run_tolerances` loops over the
    parameters. This works well where running the ``wigglefunc`` is relatively
    fast, but propagating the photons through ``instrum_before`` is slow and
    minimizes the memory footprint, because only one photon list is in memory
    at any one time. It also implies that runs for the same wiggle parameters
    but different energies are run on different realizations of any random
    draws that are performed by ``wigglefunc``.


    Parameters
    ----------
    source : `marxs.source.Source`
        Source used to generate the photons for every energy. This function
        changes the energy of the source, so the source should be set for
        monoenergetic emission with a constant flux of 1.
    energies : `astropy.units.quantity.Quantity`
        An array of energy values.
    instrum_before : `marxs.simulator.Sequence`
        An instance of the instrument which contains all elements **before**
        the first elements in ``wiggleparts``. The first element should be a
        `marxs.source.pointing.PointingModel`.  In principle,
        ``instrum_before`` can be an empty sequence and the entire instrument
        can be defined in ``instrum_remaining``, however, ``instrum_before`` is
        run only once per energy and thus it can greatly speed up the
        simulation to set this.
    instrum_remaining : `marxs.simulator.Sequence`
        An instance of the instrument which contains all elements not included
        in ``instrum_before``. ``instrum_remaining`` should include elements
        that are toleranced in this run and those that are located behind them.
    wigglefunc, wiggleparts, parameters, analyzefunc :
        These parameters are passed to
        `marxs.design.tolerancing.run_tolerances`. See that function for a
        description of these parameters.
    reset : dict or ``None``
        A dictionary of values for the ``wigglefunc`` function that resets
        the positions or properties of the wiggled elements to their default.
        If ``reset=None``, then the elements affected by ``wigglefunc`` will
        be in the state corresponding to the last entry in ``parameters`` when
        this function exits.
    t_source : int
        parameter for ``source.generate_photons()``. If the source flux is set
        to one, then ``t_source`` determines the number of photons used in the
        tolerancing simulation.

    Returns
    -------
    result : `astropy.table.Table`
        Each row in the table contains energy, wave, parameter values, and
        results from ``analyzefunc`` for a single run.

    '''
    wave = energies.to(u.Angstrom, equivalencies=u.spectral())
    outtabs = []

    for i, e in enumerate(energies):
        source.energy = e
        photons_in = source.generate_photons(t_source)
        photons_in = instrum_before(photons_in)
        data = run_tolerances(photons_in, instrum_remaining,
                              wigglefunc, wiggleparts,
                              parameters, analyzefunc)
        # convert tab into a table.
        # astropy.tables has problems with Quantities as input
        tab = Table([{d: data[i][d].value
                      if isinstance(data[i][d], u.Quantity) else data[i][d]
                      for d in data[i]} for i in range(len(data))])
        tab['energy'] = e
        tab['wave'] = wave[i]
        outtabs.append(tab)
    # Reset so that same instance of instrum can be used again
    if reset is not None:
        wigglefunc(wiggleparts, **reset)

    return table.vstack(outtabs)


def generate_6d_wigglelist(trans, rot,
                           names=['dx', 'dy', 'dz', 'rx', 'ry', 'rz']):
    '''Generate a list of parameters for the wiggle functions in this module.

    This modules contains several wiggle functions such as
    `~marxs.design.tolerances.moveglobal` or
    `~marxs.design.tolerances.wiggle` that expect input for 6 degrees of
    freedom, three translations and three rotations. Commonly, we want to
    study and dof at a time. This function helps with generating lists of
    dicts for that purpose.

    Parameters
    ----------
    trans : `astropy.units.quantity.Quantity`
        Steps for translation. The first element should be 0.
    rot : `astropy.units.quantity.Quantity`
        Steps for rotation. The first element should be 0.

    Returns
    -------
    changeglobal : list of dicts
        This list contains changes in positive and negative directions.
        Use this as input for e.g. `~marxs.design.tolerances.moveglobal`.
    changeindividual : list of dicts
        This list contains only one side, so use this as input for
        e.g. `~marxs.design.tolerances.wiggle` where the actual misalignment
        is drawn from a distribution.

    Examples
    --------
    In this example, we take small steps close to 0 and then increase the
    step size for larger distances, going up to a misalignment of 10 mm in
    linear translation and 2 degrees in rotation::

        >>> import astropy.units as u
        >>> from marxs.design.tolerancing import generate_6d_wigglelist
        >>> trans = [0., .1, .2, .4, .7] * u.cm
        >>> rot = [0., 2., 5., 10., 20] * u.arcmin
        >>> lglob, lind = generate_6d_wigglelist(trans, rot)

    '''
    if (trans.value[0]) != 0 or (rot.value[0] != 0):
        warnings.warn('First element of trans and rot should be 0.')
    n_trans = len(trans)
    n_rot = len(rot)
    n = 3 * n_trans
    changeglobal = np.zeros((n_trans * 2 * 3 + n_rot  * 2 * 3, 6))
    for i in range(3):
        changeglobal[i * n_trans: (i + 1) * n_trans, i] = trans.to(u.mm).value
        changeglobal[n + i * n_rot: n + (i + 1) * n_rot, i + 3] = rot.to(u.rad).value

    half = changeglobal.shape[0] / 2
    changeglobal[int(half):, :] = - changeglobal[:int(half), :]
    changeindividual = changeglobal[: int(half), :]
    # Remove multiple entries with [0,0,0,0, ...]
    changeglobal = np.unique(changeglobal, axis=0)
    changeindividual = np.unique(changeindividual, axis=0)

    # numpy array to list of dicts
    changeglobal = [dict(zip(names, row)) for row in changeglobal]
    changeindividual = [dict(zip(names, row)) for row in changeindividual]

    return changeglobal, changeindividual


def select_1dof_changed(table, par,
                        parlist=['dx', 'dy', 'dz', 'rx', 'ry', 'rz']):
    '''Select subset of tolerancing table with changes in 1 dof only

    This function selects a subset of a table where all parameters other
    than the parameter selected right now are zero.

    This is a helper function for analyzing and plotting results.

    Parameters
    ----------
    table : `astropy.table.Table`
        Table with wiggle results
    par : string
        Name of parameter to be selected
    parlist : list of strings
        Name of all parameters in ``table``

    Returns
    -------
    tab : `astropy.table.Table`
        Filtered table
    '''
    pars = set(parlist)
    ind = np.ones(len(table), dtype=bool)
    for p in pars - set([par]):
        ind *= table[p] == 0
    return table[ind]


def plot_wiggle(tab, par, parlist, ax, axt=None,
                R_col='Rgrat', Aeff_col='Aeffgrat',
                axes_facecolor='w'):
    '''Plotting function for overview plot wiggeling 1 dof at the time.

    For parameters starting with "d" (e.g. "dx", "dy", "dz"), the plot axes
    will be labeled as a shift, for parameters tarting with "r" as rotation.

    Parameters
    ----------
    table : `astropy.table.Table`
        Table with wiggle results
    par : string
        Name of parameter to be plotted
    parlist : list of strings
        Name of all parameters in ``table``
    ax : `matplotlib.axes.Axes`
        Axis object to plot into.
    axt : ``None`` or  `matplotlib.axes.Axes`
        If this is ``None``, twin axis are created to show resolving power
        and effective area in one plot. Alternatively, a second axes instance
        can be given here.
    R_col : string
        Column name in ``tab`` that hold the resolving power to be plotted.
        Default is set to work with `marxs.design.tolerancing.CaptureResAeff`.
    Aeff_col : string
        Column name in ``tab`` that hold the effective area to be plotted.
        Default is set to work with `marxs.design.tolerancing.CaptureResAeff`.
    axes_facecolor : any matplotlib color specification
        Color for the background in the plot.
    '''
    import matplotlib.pyplot as plt

    t = select_1dof_changed(tab, par, parlist)
    t.sort(par)
    t_wave = t.group_by('wave')
    if axt is None:
        axt = ax.twinx()

    for key, g in zip(t_wave.groups.keys, t_wave.groups):
        if par[0] == 'd':
            x = g[par]
        elif par[0] == 'r':
            x = np.rad2deg(g[par].data)
        else:
            raise ValueError("Don't know how to plot {}. Parameter names should start with 'd' for shifts and 'r' for rotations.".format(par))

        ax.plot(x, g[R_col], label='{:3.1f} $\AA$'.format(key[0]), lw=1.5)
        axt.plot(x, g[Aeff_col], ':', label='{:2.0f} $\AA$'.format(key[0]), lw=2)
    ax.set_ylabel('Resolving power (solid lines)')
    axt.set_ylabel('$A_{eff}$ [cm$^2$] (dotted lines)')
    if par[0] == 'd':
        ax.set_xlabel('shift [mm]')
        ax.set_title('Shift along {}'.format(par[1]))
    elif par[0] == 'r':
        ax.set_xlabel('Rotation [degree]')
        ax.set_title('Rotation around {}'.format(par[1]))

    for a in [ax, axt]:
        a.set_facecolor(axes_facecolor)
        a.set_axisbelow(True)
        a.grid(axis='x', c='1.0', lw=2, ls='solid')


wiggle_plot_facecolors = {'global': '0.9',
                          'individual': (1.0, 0.9, 0.9)}
'''Default background colors for wiggle overview plots.

If the key of the dict matches part of the filename, the color listed in
the dict is applied.
'''


def load_and_plot(filename, parlist=['dx', 'dy', 'dz', 'rx', 'ry', 'rz'], **kwargs):
    '''Load a table with wiggle results and make default plot

    This is a function to generate a quicklook image with many
    hardcoded defaults for figure size, colors etc.
    In particular, this function is written for the display of
    6d plots which vary 6 degrees of freedom, one at a time.

    The color for the background in the plot is set depending on the filename
    using the ``string : color`` assignments in
    `~marxs.design.tolerancing.wiggle_plot_facecolors`. No fancy regexp based
    match is applied, this is simply a check with ``in``.

    Parameters
    ----------
    filename : string
        Path to a file with data that can be plotted by
        `~marxs.design.tolerancing.plot_wiggle`.

    parlist : list of strings
        Name of all parameters in ``table``.
        This function only plots six of them.

    Returns
    -------
    tab : `astropy.table.Table`
        Table of data read from ``filename``
    fig : `matplotlib.figure.Figure`
        Figure with plot.
    kwargs :
        All other parameters are passed to
        `~marxs.design.tolerancing.plot_wiggle`.
    '''
    import matplotlib.pyplot as plt

    tab = Table.read(filename)

    if 'axis_facecolor' not in kwargs:
        for n, c in wiggle_plot_facecolors.items():
            if n in filename:
                kwargs['axes_facecolor'] = c

    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(wspace=.6, hspace=.3)
    for i, par in enumerate(parlist):
        ax = fig.add_subplot(2, 3, i + 1)
        plot_wiggle(tab, par, parlist, ax, **kwargs)

    return tab, fig
