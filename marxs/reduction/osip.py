# Licensed under GPL version 3 - see LICENSE.rst
from os.path import join as pjoin
import logging
import os
from abc import ABC
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from . import ogip
from marxs.analysis.plots import OrderColor


logger = logging.getLogger(__name__)

__all__ = ['OSIPBase',
           'FixedWidthOSIP', 'FixedFractionOSIP', 'FractionalDistanceOSIP',
           ]


class OSIPBase(ABC):
    """Modify ARF files to account for order-sorting effects

    This is a base class that implements order sorting and integration of
    the probabilities (OSIP). This includes order-sorting of a photon list
    (not yet implemented) and  methods to modify ARF files
    to account for order-sorting. Different diffraction orders fall on
    the same physical space on the CCD. The CCD energy resolution can
    then be used to assign photons to specific grating
    orders. However, since the CCD energy resolution is finite, this
    assignment is not perfect. Some photons may fall outside of the
    nominal energy used for order sorting and thus be lost, other may
    fall onto the energy assigned to the different order. This class
    modifies ARF files to correct for the integrated probability that a photon
    falls inside the order-sorting region.

    Derived classes should override either `osip_tab` or `osip_range`. The
    other of the two can always be calculated from whatever one is given.

    Parameters
    ----------
    offset_orders : list of int
        Offset from main order that is relevant.

    ccd_redist : Redistribution object
        Function that return the width the Gaussian sigma of the CCD
        resolution, given an input energy.
        TODO: Better docs here
    """
    osip_description = 'Not implemented'
    '''String to be added to FITS header in OSIP keyword for ARFs written.'''

    def __init__(self, ccd_redist, offset_orders=[-1, 0, 1],
                 filename_from_meta=ogip.filename_from_meta):
        self.offset_orders = offset_orders
        self.ccd_redist = ccd_redist
        self.filename_from_meta = filename_from_meta

    @u.quantity_input(chan_mid=u.keV, equivalencies=u.spectral())
    def osip_tab(self, chan_mid, order) -> u.eV:
        '''Calculate the boundaries of an order-sorting region

        Parameters
        ----------
        energy : `~astropy.units.quantity.Quantity`
            Energy (or wavelength actually) for which the OSIP should
            be calculated, e.g. the mid-point of a channel
        order : int
            Diffraction order.

        Returns
        -------
        osip_tab : `~astropy.units.quantity.Quantity`
            The shape is an array that contains, for each point in `energy`,
            the width of the OSIP region, as measured from the nominal
            energy. If, for example, the energy is 1 keV and ``osip_tab``
            is `[.1, .2] * u.keV`, then the osip is [.9, 1.2] keV.
        '''
        osiprange = self.osip_range(chan_mid, order)
        osiprange = osiprange.to(u.eV, equivalencies=u.spectral())
        E = chan_mid.to(u.eV, equivalencies=u.spectral())
        return E + osiprange * np.array([-1, 1])[:, np.newaxis]

    @u.quantity_input(chan_mid=u.keV, equivalencies=u.spectral())
    def osip_range(self, chan_mid, order) -> u.eV:
        '''Calculate the boundaries of an order-sorting region

        This method returns the boundaries of an order-sorting region in
        energy space.
        This is very similar to `osip_tab`, which gives the width of the
        region. Essentially, this method calls `osip_tab` and add the energy
        so that the result contains the lower and upper bound.

        Parameters
        ----------
        chan_mid : `~astropy.units.quantity.Quantity`
            Energy or wavelength for which the OSIP should
            be calculated, e.g. the mid-point of a channel
        order : int
            Diffraction order.

        Returns
        -------
        osip : `~astropy.units.quantity.Quantity`
            The shape is an array that contains, for each point in `energy`,
            the lower and upper bound of the OSIP region.
        '''
        osiptab = self.osip_tab(chan_mid, order)
        osiptab = osiptab.to(u.eV, equivalencies=u.spectral())
        E = chan_mid.to(u.eV, equivalencies=u.spectral())
        return E + osiptab * np.array([-1, 1])[:, np.newaxis]

    @u.quantity_input(chan_mid_nominal=u.keV, equivalencies=u.spectral())
    def osip_factor(self, chan_mid_nominal, o_nominal, o_true):
        '''Calculate the relative effective area after order sorting.

        This method can be used to calculate a fraction fo

        Parameters
        ----------
        chan_mid_nominal : `~astropy.units.quantity.Quantity`
            Energy or wavelength of the nominal order for which the OSIP should
            be calculated, e.g. the mid-point of a channel
        o_nominal : int
            Nominal diffraction order, i.e. the order that is to be extracted
            from the CCD.
        o_true : int
            True diffraction order.

        Returns
        -------
        osip : `~astropy.units.quantity.Quantity`
            The shape is an array that contains, for each point in `energy`,
            the lower and upper bound of the OSIP region.
        '''

        if np.sign(o_nominal) != np.sign(o_true):
            return 0.

        osiprange = self.osip_range(chan_mid_nominal, o_nominal)
        Etrue = chan_mid_nominal.to(u.eV, equivalencies=u.spectral()) * \
            (o_true / o_nominal)
        upper = self.ccd_redist.cdf(osiprange[1, :], loc=Etrue)
        lower = self.ccd_redist.cdf(osiprange[0, :], loc=Etrue)
        return upper - lower

    def apply_osip(self, inputarf, outpath, order, outroot='',
                   overwrite=False):
        '''Modify an ARF to account for incomplete order sorting

        This function reads an input ARF file, which contains the
        effective area for a grating order in Arcus. For a given order
        sorting window, it then calculates what fraction of the
        photons is lost. For example, if the order-sorting regions can be
        is chosen to contain  90% energy fraction, then the new ARF values
        will be 0.9 (the integrated probability over the order-sorting region)
        times the input ARF.

        If the ``order`` of the new ARF differs from the order of the
        input ARF, then the new ARF is representative of order
        confusion, e.g. is shows how many photons of order 4 are
        sorted into the order=5 grating spectrum.

        Parameters
        ----------
        inputarf : string
            Filename and path of input arf
        outpath : string
            Location where the output arfs are deposited
        order : int
            Nominal order. (The true order of the input arf is taken from the
            input arf header data.)
        outfile : string
            path and filename for output file
        overwrite : bool
            Overwrite existing files?
        '''
        arf = ogip.ARF.read(inputarf)
        try:
            arf['SPECRESP'] = arf['SPECRESP'] / arf['OSIPFAC']
            logger.info(f'{inputarf} already has OSIP applied, reverting ' +
                        'before applying new OSIP.')
        except KeyError:
            pass

        m = int(arf.meta['ORDER'])

        energies = 0.5 * (arf['ENERG_LO'] + arf['ENERG_HI'])
        osip_fac = self.osip_factor(energies / m * order, order, m)
        arf['SPECRESP'] = osip_fac * arf['SPECRESP']
        arf['OSIPFAC'] = osip_fac

        arf.meta['INSTRUME'] = f'ORDER_{order}'
        arf.meta['OSIP'] = self.osip_description
        arf.meta['TRUEORD'] = f'{m}'
        arf.meta['CCDORDER'] = f'{order}'
        # some times there is no overlap and all elements become 0
        if np.all(arf['SPECRESP'] == 0):
            logger.warning(f'True refl order {m} does not contribute to ' +
                        f'CCDORDER {order}. ' +
                        'Writing ARF with all entries equal to zero.')
        os.makedirs(outpath, exist_ok=True)
        arf.write(pjoin(outpath, outroot +
                        self.filename_from_meta('arf', **arf.meta)),
                  overwrite=overwrite)


    def plot_osip(self, ax, grid, order, **kwargs):
        '''Plot banana plot with OSIP region marked.

        Parameters
        ----------
        ax : `matplotlib.axes._subplots.AxesSubplot`
            The axes into which the banana is plotted.
        grid : `~astropy.units.quantity.Quantity`
            Wavelength grid in m lambda
        order : int
            Order number
        kwargs :
            Any other parameters are passed to ``plt.plot``
        '''
        grid = grid.to(u.Angstrom, equivalencies=u.spectral())
        en = (grid / np.abs(order)).to(u.keV, equivalencies=u.spectral())

        ohw = self.osip_tab(en, order)

        line = ax.plot(grid, en, label=order, **kwargs)
        ax.fill_between(grid.value,
                        (en - ohw[0, :]).value,
                        (en + ohw[1, :]).value,
                        color=line[0].get_color(), alpha=.2,
                        label='__no_legend__')
        ax.set_xlabel(f'$m\\lambda$ [{grid.unit.to_string("latex_inline")}]')
        ax.set_ylabel('CCD energy [keV]')
        ax.set_xlim([grid.value.min(), grid.value.max()])
        ax.legend()
        ax.set_title('Order sorting regions')

    def plot_mixture(self, ax, grid, order):
        '''Plot relative contribution of main order and interlopers.

        Parameters
        ----------
        ax : `matplotlib.axes._subplots.AxesSubplot`
            The axes into which the lines are plotted.
        grid : `~astropy.units.quantity.Quantity`
            Wavelength grid in m lambda
        order : int
            Order number
        '''
        grid = grid.to(u.Angstrom, equivalencies=u.spectral())
        ax.axhspan(1, 2, facecolor='r', alpha=.3, label='extractions overlap')
        en = (grid / np.abs(order)).to(u.keV, equivalencies=u.spectral())
        cm = self.osip_factor(en, order, order)
        ax.plot(grid, cm, label='main order', lw=2)
        for o in self.offset_orders:
            if o == 0:
                continue
            coffset = self.osip_factor(en, order, order + o)
            ax.plot(grid, coffset, label=f'contam {o}',
                    # Different linestyle to avoid overlapping lines in plot
                    ls={-1: '-', +1: ':'}[np.sign(o)]
                    )
            cm += coffset
        ax.plot(grid, cm, 'k', label='sum', lw=3)
        ax.set_xlabel(f'$m\\lambda$ [{grid.unit.to_string("latex_inline")}]')
        ax.set_ylabel('Fraction of photons in OSIP')
        ax.set_title(f'Order {order}')
        ax.set_xlim([grid.value.min(), grid.value.max()])
        ax.set_ylim(0, cm.max() * 1.05)
        ax.legend()

    def plot_summary(self, inputarf, orders, outpath, outroot=''):
        '''Write summary plot to directory with ARFs

        Parameters
        ----------
        inputarf : string
            Path to one input ARF. The energy grid for the plot
            is taken from that ARF.
        outpath : string
            Location where the output ARFs are deposited
        outroot : string
            prefix for output filename
        '''
        arf = ogip.ARF.read(inputarf)
        bin_mid = 0.5 * (arf['ENERG_LO'] + arf['ENERG_HI'])
        grid = bin_mid.to(u.Angstrom, equivalencies=u.spectral())
        fig, axes = plt.subplots(ncols=2, figsize=(8, 4))

        oc = OrderColor(max_order=np.max(np.abs(orders)))

        for order in orders:
            self.plot_osip(axes[0], grid * abs(arf.meta['ORDER']), order,
                           **oc(order))
        # pick the middle order for plotting purposes
        o_mid = orders[len(orders) // 2]
        self.plot_mixture(axes[1], grid * abs(arf.meta['ORDER']), o_mid)
        fig.subplots_adjust(wspace=.3)
        fig.savefig(pjoin(outpath, outroot + 'OSIP_regions.pdf'),
                    bbox_inches='tight')

    def plot_summed_arf(self, path):
        '''Generate a plot of the summed effective area.

        This plot is not meant to convey detailed information, but rather
        serve as a check for verification.

        Parameters
        ----------
        path : string
            Path to location of ARFs. The plot is written to the same file.
        '''
        wavegrid = np.arange(0, 60, .1) * u.Angstrom
        specresp = np.zeros(len(wavegrid)) * u.cm**2
        arflist = glob(pjoin(path, '*.arf'))
        for f in arflist:
            arf = ogip.ARF.read(f)
            # Interpolate on common wavelength grid. Good enough for plotting.
            # Interp requires sorted input
            # but ARFs are reverse-sorted in wavelength
            specresp += np.interp(wavegrid, arf['BIN_LO'][::-1],
                                  arf['SPECRESP'][::-1])

        fig, ax = plt.subplots()
        ax.plot(wavegrid, specresp)
        ax.set_xlabel(f'$\\lambda$ [{wavegrid.unit.to_string("latex_inline")}]')
        ax.set_ylabel(f'effective area [{specresp.unit.to_string("latex_inline")}]')
        ax.set_title('Combined effective area of all dispersed orders')
        fig.savefig(pjoin(path, 'ARF_sum.pdf'),
                    bbox_inches='tight')

    def apply_osip_all(self, inpath, outpath, orders,
                       inroot='', outroot='',
                       overwrite=False,
                       filename_from_meta_kwargs={}):
        '''Apply OSIP to many arfs at once

        This routine iterates over orders and offset orders to produce
        arfs that describe the contamination due to insufficient order
        sorting.  When a single order is extracted (e.g. order 5), but
        the CCD resolution is insufficient to really separate the
        orders well, then some photons from order 4 and 6 might end up
        in the extracted spectrum. This function generates the
        appropriate arfs.

        Input arfs need to follow the arcus filename convention and
        all be located in the same directory. In addition to the ARFs
        with order-sorting applied, this method also places an
        overview plot in the output directory.

        Parameters
        ----------
        inpath : string
            Directory with input ARFs.
        outpath : string
            Location where the output ARFs are deposited
        orders : list of int
            Nominal CCD orders to be processed
        inroot : string
            prefix for input filename
        outroot : string
            prefix for output filename
        overwrite : bool
            Overwrite existing files?
        filename_from_meta_kwargs : dict
            Any additional kwargs needed to run ``self.filename_from_meta``
            and thus pick the correct input arfs in the `inpath`
        '''
        goodarf = None
        for order in orders:
            for t in self.offset_orders:
                # No contamination by zeroth order or by orders on the other
                # side of the zeroth order
                if (order + t != 0) and (np.sign(order) == np.sign(order + t)):
                    inputarf = pjoin(inpath, inroot +
                                     self.filename_from_meta(filetype='arf',
                                                        ORDER=order + t,
                                                        **filename_from_meta_kwargs))
                    try:
                        self.apply_osip(inputarf, outpath, order,
                                        outroot=outroot, overwrite=overwrite)
                        goodarf = inputarf
                    except FileNotFoundError:
                        logger.info(f'Skipping order: {order}, offset: {t} ' +
                                    'because input arf not found')
                        continue

        # The second condition checks that at least one ARF was written
        if goodarf is not None:
            self.plot_summary(goodarf, orders, outpath, outroot)
            self.plot_summed_arf(outpath)


class FixedWidthOSIP(OSIPBase):
    """Modify ARF files to account for order-sorting effects

    This class implements order sorting and integration of
    the probabilities (OSIP) for order-sorting regions that have a fixed witdh
    in energy.
    This includes order-sorting of a photons list
    (not yet implmented) and  methods to modify ARF files
    to account for order-sorting. Different diffraction orders fall on
    the same physical space on the CCD. The CCD energy resolution can
    then be used to assign photons to specific grating
    orders. However, since the CCD energy resolution is finite, this
    assingment is not perfect. Some photons may fall outside of the
    nominal energy used for order sorting and thus be lost, other may
    fall onto the energy assgined tothe different order. This class
    modifies ARF files to correct for the integrated probability that a photon
    falls inside the order-sorting region.

    Parameters
    ----------
    halfwidth : `astropy.units.quantity.Quantity`
        Half-wdith of the order sorting region. The same width is used for all
        wavelength.
    offset_orders : list of int
        Offset from main order that is relevant.
    ccd_redist : callable
        Function that returns the width the Gaussian sigma of the CCD
        resolution, given an input energy.
    """
    def __init__(self, halfwidth, **kwargs):
        self.halfwidth = halfwidth
        super().__init__(**kwargs)

    @property
    def osip_description(self):
        return str(self.halfwidth)

    @u.quantity_input(chan_mid_nominal=u.keV, equivalencies=u.spectral())
    def osip_tab(self, chan_mid_nominal, order):
        return np.broadcast_to(self.halfwidth,
                               (2, len(chan_mid_nominal)), subok=True)


class FixedFractionOSIP(OSIPBase):
    '''Modify ARF files to account for order-sorting effects

    This class implements order sorting and integration of
    the probabilities (OSIP) for order sorting regions that contain a fixed
    fraction of the photons in energy space.
    This includes order-sorting of a photons list
    (not yet implmented) and  methods to modify ARF files
    to account for order-sorting. Different diffraction orders fall on
    the same physical space on the CCD. The CCD energy resolution can
    then be used to assign photons to specific grating
    orders. However, since the CCD energy resolution is finite, this
    assingment is not perfect. Some photons may fall outside of the
    nominal energy used for order sorting and thus be lost, other may
    fall onto the energy assgined tothe different order. This class
    modifies ARF files to correct for the integrated probability that a photon
    falls inside the order-sorting region.

    Parameters
    ----------
    fraction : float
        Number (between 0 and 1) that determins which fraction of the CCD
        energy distriution should be covered by the order sorting regions.
        The same width is used for all
        wavelength.
    offset_orders : list of int
        Offset from main order that is relevant.
    ccd_redist : callable
        Function that return the width the Gaussian sigma of the CCD
        resolution, given an input energy.
    '''
    def __init__(self, fraction, **kwargs):
        self.fraction = fraction
        super().__init__(**kwargs)

    @property
    def osip_description(self):
        return 'OSIPFrac' + str(self.fraction)

    @u.quantity_input(chan_mid_nominal=u.keV, equivalencies=u.spectral())
    def osip_range(self, chan_mid_nominal, order):
        return self.ccd_redist.interval(self.fraction, loc=chan_mid_nominal)


class FractionalDistanceOSIP(OSIPBase):
    '''Modify ARF files to account for order-sorting effects

    This class implements order sorting and integration of
    the probabilities (OSIP) for order sorting regions that fill a fixed
    fraction of the energy space.
    This includes order-sorting of a photons list
    (not yet implmented) and  methods to modify ARF files
    to account for order-sorting. Different diffraction orders fall on
    the same physical space on the CCD. The CCD energy resolution can
    then be used to assign photons to specific grating
    orders. However, since the CCD energy resolution is finite, this
    assingment is not perfect. Some photons may fall outside of the
    nominal energy used for order sorting and thus be lost, other may
    fall onto the energy assgined tothe different order. This class
    modifies ARF files to correct for the integrated probability that a photon
    falls inside the order-sorting region.

    Parameters
    ----------
    fraction : float
        Fraction (between 0 and 1) of the space between orders that will be
        covered by the extration region. For a value of 1, order-sorting
        regions just touch and each photon will be assigned to exactly one
        order.
    offset_orders : list of int
        Offset from main order that is relevant.
    ccd_redist : callable
        Function that return the width the Gaussian sigma of the CCD
        resolution, given an input energy.
    '''
    def __init__(self, fraction=1., **kwargs):
        self.fraction = fraction
        super().__init__(**kwargs)

    @property
    def osip_description(self):
        return 'OSIPDist' + str(self.fraction)

    @u.quantity_input(chan_mid_nominal=u.keV, equivalencies=u.spectral())
    def osip_tab(self, chan_mid_nominal, order):
        energy = chan_mid_nominal.to(u.keV, equivalencies=u.spectral())
        dE = energy / abs(order)
        return np.broadcast_to(self.fraction / 2 * dE,
                               (2, len(chan_mid_nominal)), subok=True)
