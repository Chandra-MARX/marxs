# Licensed under GPL version 3 - see LICENSE.rst
'''
Sources generate photons with all photon properties: energy, direction, time, polarization.

For objects of type `AstroSource`, the coordinates of the photon origin on the sky are added to the photon list. `astropy.coords.SkyCoord` is an object well suited for this task. These objects can be added to photon tables through the mechanims of `mixin columns <http://docs.astropy.org/en/latest/table/index.html#mixin-columns>`). However, mix-in columns don't (yet) support saving to all table formats or tables operations such as stacking. Thus, it is much better to include the coordinates as two columns of floats with names ``ra`` and ``dec`` into the table.

Sources take a `~astropy.coordinates.SkyCoord` from the user to avoid any ambiguity about the coordinate systme used, but convert this into plain floats used in the photon table.
'''

import os
from datetime import datetime

import numpy as np
from scipy.stats import expon
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame

from ..base import SimulationSequenceElement
from ..math.random import RandomArbitraryPdf
from .. import __version__ as marxsversion


@u.quantity_input
def poisson_process(rate: (u.s * u.cm**2)**(-1)):
    '''Return a function that generates Poisson distributed times with rate ``rate``.

    Parameters
    ----------
    rate :  `~astropy.units.quantity.Quantity`
        Expectation value for the total rate of photons with unit 1 / cm**2 / s.

    Returns
    -------
    poisson_rate : function
        Function that generates Poisson distributed times with rate ``rate``.
    '''
    if not rate.isscalar:
        raise ValueError('"rate" must be scalar.')

    @u.quantity_input(exposuretime=u.s)
    def poisson_rate(exposuretime: u.s, geomarea: u.cm**2) -> u.s:
        '''Generate Poisson distributed times.

        Parameters
        ----------
        exposuretime : `~astropy.units.quantity.Quantity`
            Exposure time
        geomarea : `~astropy.units.quantity.Quantity`
            Geometric opening area of telescope

        Returns
        -------
        times :  `~astropy.units.quantity.Quantity`
            Poisson distributed times.
        '''
        fullrate = (rate * geomarea).to(u.s**(-1)).value
        # Make 10 % more numbers then we expect to need, because it's random
        times = expon.rvs(scale=1./fullrate,
                          size=int(exposuretime.value * fullrate * 1.1))
        # If we don't have enough numbers right now, add some more.
        while (times.sum() * u.s) < exposuretime:
            times = np.hstack([times, expon.rvs(scale=1/fullrate,
                                                size=int((exposuretime.to(u.s).value - times.sum()) * fullrate * 1.1))])
        times = np.cumsum(times)
        return times[times < exposuretime.value] * u.s
    return poisson_rate


class SourceSpecificationError(Exception):
    pass


class Source(SimulationSequenceElement):
    '''Base class for all photons sources.

    This class provides a very general implementation of photons
    sources. Typically, it is not used directly, but a more specialized
    subclass, such as `PointSource` for an astronomical source or
    `LabPointSource` for a source at a finite distance.

    Most of the derived source support the same input argumets as `Source`,
    thus they are explained in detail here.

    Parameters
    ----------
    flux : `~astropy.units.quantity.Quantity` or callable
         This sets the total flux from a source in
        photons/time/area.  Options are:

        - quantity: Constant (not Poisson distributed) flux.
        - callable: Function that takes a total exposure time as input and
          returns an array
          of photon emission times between 0 and the total exposure time.

    energy : polarization.
        - `~astropy.units.quantity.Quantity` of callable or `astropy.table.Table`

        This input decides the energy of the emitted photons.
        Possible formats are:

        - polarization.
        - `~astropy.units.quantity.Quantity`: Constant energy.
        - `astropy.table.Table`:
          Given this table, the code assumes a piecewise flat spectrum.
          The "energy" values contain the **upper** limit of each bin,
          the "fluxdensity" array the flux density in each bin.
          The first entry in the "fluxdensity" array is ignored, because the lower
          bound of this bin is undefined.
          The code draws an energy from this spectrum for every photon created.
        - A function or callable object: This option allows for full
          customization. The function must take an array of photon times as
          input and return an equal length array of photon energies
          `~astropy.units.quantity.Quantity`.

    polarization : `~astropy.units.quantity.Quantity` or ``None``,  `astropy.table.Table` or callable.
        There are several different ways to set the polarization angle of the
        photons for a polarized source. In all cases, the angle is measured
        North through East. (We ignore the special case of a polarized source
        exactly on a pole.)
        The default value is ``None`` (unpolarized source).

        - ``None`` : An unpolarized source. Every photons is assigned a random
          polarization.
        - `~astropy.units.quantity.Quantity` :
          Constant polarization angle for all photons
        - `~astropy.table.Table` :
          Table with columns called "angle" and "probabilitydensity".
          The summed probability density will automatically be
          normalized to one. Given this table, the
          code assumes a piecewise constant probability density. The "angle"
          values contain the **upper** limit of each bin. The first entry in the
          "probabilitydenisty" array is ignored, because the lower bound of this bin
          is undefined.
        - a callable (function or callable object): This
          option allows full customization.  The function is called with two
          arrays (time and energy values) as input and must return an array of
          equal length that contains the polarization angles as
          `~astropy.units.quantity.Quantity` object.

    geomarea : `astropy.units.Quantity` or ``None``:
          Geometric opening area of telescope. If ``None`` then the flux must
          be given in photons per time, not per time per unit area.

    '''
    def __init__(self, energy=1*u.keV, flux=1 / u.s / u.cm**2,
                 polarization=None, geomarea=1*u.cm**2, **kwargs):
        self.energy = energy
        self.flux = flux
        self.polarization = polarization
        self.geomarea = 1 if geomarea is None else geomarea

        super(Source, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.generate_photons(*args, **kwargs)

    @u.quantity_input()
    def generate_times(self, exposuretime: u.s):
        if callable(self.flux):
            return self.flux(exposuretime, self.geomarea)
        elif hasattr(self.flux, 'isscalar') and self.flux.isscalar:
            return np.arange(0, exposuretime.to(u.s).value,
                             1. / (self.flux * self.geomarea * u.s).decompose()) * u.s
        else:
            raise SourceSpecificationError('`flux` must be a quantity or a callable.')

    @u.quantity_input()
    def generate_energies(self, t: u.s) -> u.keV:
        n = len(t)
        # function
        if callable(self.energy):
            en = self.energy(t)
            if len(en) != n:
                raise SourceSpecificationError('`energy` has to return an array of same size as input time array.')
            else:
                return en
        # astropy.table.QTable
        elif hasattr(self.energy, 'columns'):
            x = self.energy['energy'].to(u.keV).value
            y = (self.energy['fluxdensity'][1:] * np.diff(self.energy['energy'])).to((u.s * u.cm**2)**(-1)).value
            y = np.hstack(([0], y))
            rand = RandomArbitraryPdf(x, y)
            return rand(n) * u.keV
        # scalar quantity
        elif hasattr(self.energy, 'isscalar') and self.energy.isscalar:
            return np.ones(n) * self.energy.to(u.keV,
                                               equivalencies=u.spectral())
        # anything else
        else:
            raise SourceSpecificationError('`energy` must be Quantity, function, or have columns "energy" and "fluxdensity".')

    @u.quantity_input()
    def generate_polarization(self, times: u.s, energies: u.keV) -> u.rad:
        n = len(times)
        # function
        if callable(self.polarization):
            pol = self.polarization(times, energies)
            if len(pol) != n:
                raise SourceSpecificationError('`polarization` has to return an array of same size as input time and energy arrays.')
            else:
                return pol
        elif self.polarization is None:
            return np.random.uniform(0, 2 * np.pi, n) * u.rad
        # astropy.table.QTable
        elif hasattr(self.polarization, 'columns'):
            x = self.polarization['angle'].to(u.rad).value
            y = (self.polarization['probabilitydensity'][1:] * np.diff(self.polarization['angle'])).to(u.dimensionless_unscaled).value
            y = np.hstack(([0], y))
            rand = RandomArbitraryPdf(x, y)
            return rand(n) * u.rad
        # scalar quantity
        elif hasattr(self.polarization, 'isscalar') and self.polarization.isscalar:
            return np.ones(n) * self.polarization
        else:
            raise SourceSpecificationError('`polarization` must be number (angle), callable, None (unpolarized), 2.n array or have fields "angle" (in rad) and "probability".')

    def generate_photon(self):
        raise NotImplementedError

    @u.quantity_input()
    def generate_photons(self, exposuretime: u.s):
        '''Central function to generate photons.

        Calling this function generates a photon table according to
        the `flux`, `energy`, and `polarization` of this source. The
        number of photons depends on the total exposure time, which is
        a parameter of this function. Depending on the setting for
        `flux` the photons could be distributed equally over the
        interval 0..exposuretime or follow some other distribution.

        Parameters
        ----------
        exposuretime : `astropy.quantity.Quantity`
            Total exposure time.

        Returns
        -------
        photons : `astropy.table.Table`
            Table with photon properties.

        '''
        times = self.generate_times(exposuretime)
        energies = self.generate_energies(times)
        pol = self.generate_polarization(times, energies)
        n = len(times)
        photons = Table([times.to(u.s).value,
                         energies.to(u.keV).value,
                         pol.to(u.rad).value,
                         np.ones(n)],
                        names=['time', 'energy', 'polangle', 'probability'])
        photons.meta['EXTNAME'] = 'EVENTS'
        photons.meta['EXPOSURE'] = (exposuretime.to(u.s),
                                    'total exposure time [s]')

        #photons.meta['DATE-OBS'] =
        photons.meta['CREATOR'] = 'MARXS - Version {0}'.format(marxsversion)
        photons.meta["LONGSTRN"] = ("OGIP 1.0", "The OGIP long string convention may be used.")
        photons.meta['MARXSVER'] = (marxsversion, 'MARXS version')
        now = datetime.now()
        photons.meta['SIMDATE'] = (str(now.date()), 'Date simulation was run')
        photons.meta['SIMTIME'] = (str(now.time())[:10], 'Time simulation was started')
        photons.meta['SIMUSER'] = (os.environ.get('USER', 'unknown user'),
                                   'User running simulation')
        photons.meta['SIMHOST'] = (os.environ.get('HOST', 'unknown host'),
                                   'Host system running simulation')
        photons['time'].unit = u.s
        photons['energy'].unit = u.keV
        photons['polangle'].unit = u.rad
        return photons


class AstroSource(Source):
    '''Astrophysical source with a sky position

    Parameters
    ----------
    coords : `astropy.coordinates.SkySoord` (preferred)
        Position of the source on the sky. If ``coords`` is not a
        `~astropy.coordinates.SkyCoord` object itself, it is used to
        initialize such an object. See `~astropy.coordinates.SkyCoord`
        for a description of allowed input values.
    '''
    def __init__(self, **kwargs):
        coords = kwargs.pop('coords')
        if isinstance(coords, SkyCoord):
            self.coords = coords
        else:
            self.coords = SkyCoord(coords)

        if not self.coords.isscalar:
            raise ValueError("Coordinate must be scalar, not array.")
        super(AstroSource, self).__init__(**kwargs)

    def set_pos(self, photons, coo):
        '''Set Ra, Dec of photons in table

        This function write Ra, Dec to a table. It is defined here to
        make the way `astropy.coordinates.SkyCoord` objects are stored
        more uniform.  Currently, mixin columns in tables have some
        disadvantages, e.g. they cause errors on writing and on
        stacking. Thus, we store the coordinates as plain
        numbers. Since that format is not unique (e.g. units could be
        deg or rad), system could be ICRS, FK4, FK5 or other this
        conversion is done here for all astrononimcal sources.  This
        also makes it easier to change that design in the future.

        Parameters
        ----------
        photons : `astropy.table.Table`
            Photon table. Columns ``ra`` and ``dec`` will be added or overwritten.
        coo : `astropy.coords.SkyCoord`
            Photon coordinates

        '''
        photons['ra'] = coo.icrs.ra.deg
        photons['dec'] = coo.icrs.dec.deg
        photons['ra'].unit = u.degree
        photons['dec'].unit = u.degree
        photons.meta['COORDSYS'] = ('ICRS', 'Type of coordinate system')


class PointSource(AstroSource):
    '''Astrophysical point source.

    Parameters
    ----------
    kwargs : see `Source`
        Other keyword arguments include ``flux``, ``energy`` and ``polarization``.
        See `Source` for details.
    '''
    def __init__(self, **kwargs):
        super(PointSource, self).__init__(**kwargs)

    @u.quantity_input
    def generate_photons(self, exposuretime: u.s):
        photons = super(PointSource, self).generate_photons(exposuretime)
        self.set_pos(photons, self.coords)
        return photons


class RadialDistributionSource(AstroSource):
    ''' Base class for sources where photons follow some radial distribution on the sky

    Parameters
    ----------
    radial_distribution : callable
        A function that takes an interger as input, which specifies the number
        of photons to produce. The output must be an `astropy.units.Quantity`` object
        with n angles in it.
    func_par : object
        ``radial_distribution`` has access to ``self.func_par`` to hold function
        parameters. This could be, e.g. a tuple with coeffications.
    kwargs : see `Source`
        Other keyword arguments include ``flux``, ``energy`` and ``polarization``.
        See `Source` for details.
    '''
    def __init__(self, **kwargs):
        self.func = kwargs.pop('radial_distribution')
        self.func_par = kwargs.pop('func_par', None)
        super(RadialDistributionSource, self).__init__(**kwargs)

    @u.quantity_input
    def generate_photons(self, exposuretime: u.s):
        '''Photon positions are generated in a frame that is centered on the
        coordinates set in ``coords``, then they get transformed into the global sky
        system.
        '''
        photons = super(RadialDistributionSource, self).generate_photons(exposuretime)

        relative_frame = SkyOffsetFrame(origin=self.coords)
        n = len(photons)
        phi = np.random.rand(n) * 2. * np.pi * u.rad
        d = self.func(n)
        relative_coords = SkyCoord(d * np.sin(phi), d * np.cos(phi), frame=relative_frame)
        origin_coord = relative_coords.transform_to(self.coords)
        self.set_pos(photons, origin_coord)

        return photons


class SphericalDiskSource(RadialDistributionSource):
    '''Astrophysical source with the shape of a circle or ring.

    The `DiskSource` makes a small angle approximation. In contrast, this source
    implements the full spherical geometry at the cost of running slower.
    For radii less than a few degrees the difference is negligible and we recommend
    use of the faster `DiskSource`.

    Parameters
    ----------
    a_inner, a_outer : `astropy.coordinates.Angle`
        Inner and outer angle of the ring (e.g. in arcsec).
        The default is a disk with no inner hole (``a_inner`` is set to zero.)
    '''
    def __init__(self, **kwargs):
        kwargs['func_par'] = [kwargs.pop('a_outer'),
                              kwargs.pop('a_inner', 0. * u.rad)]
        kwargs['radial_distribution'] = self.radial_distribution
        super(SphericalDiskSource, self).__init__(**kwargs)

    def radial_distribution(self, n):
        '''Radial distribution function.

        See http://6degreesoffreedom.co/circle-random-sampling/ for an explanation
        of how to derive the formula. Note however, that here we use sin instead of cos
        because we measure the angle from the top.
        '''
        u = np.random.rand(n)
        return np.arccos(np.cos(self.func_par[1]) * (1. - u) + u * np.cos(self.func_par[0]))


class DiskSource(RadialDistributionSource):
    '''Astrophysical source with the shape of a circle or ring.

    This source uses a small angle approximation which is valid for radii less than
    a few degrees and runs much faster. See ``SphericalDiskSource`` for an
    implementation using full spherical geometry.

    Parameters
    ----------
    a_inner, a_outer : `astropy.coordinates.Angle`
        Inner and outer angle of the ring (e.g. in arcsec).
        The default is a disk with no inner hole (``a_inner`` is set to zero.)
    '''
    def __init__(self, **kwargs):
        kwargs['func_par'] = [kwargs.pop('a_outer'),
                              kwargs.pop('a_inner', 0. * u.rad)]
        kwargs['radial_distribution'] = lambda n: np.sqrt(self.func_par[1]**2 +
                          np.random.rand(n) * (self.func_par[0]**2 - self.func_par[1]**2))
        super(DiskSource, self).__init__(**kwargs)


class GaussSource(AstroSource):
    '''Astrophysical source with a Gaussian brightness profile.

    This source uses a small angle approximation which is valid for radii less than
    a few degrees.

    Parameters
    ----------
    sigma : `astropy.coordinates.Angle`
        Gaussian sigma setting the width of the distribution
    '''
    def __init__(self, **kwargs):
        self.sigma = kwargs.pop('sigma')
        super(GaussSource, self).__init__(**kwargs)

    @u.quantity_input()
    def generate_photons(self, exposuretime: u.s):
        '''Photon positions are generated in a frame that is centered on the
        coordinates set in ``coords``, then they get transformed into the global sky
        system.
        '''
        photons = super(GaussSource, self).generate_photons(exposuretime)

        relative_frame = SkyOffsetFrame(origin=self.coords)
        n = len(photons)
        relative_coords = SkyCoord(np.random.normal(scale=self.sigma.value, size=n) * self.sigma.unit,
                                   np.random.normal(scale=self.sigma.value, size=n) * self.sigma.unit,
                                   frame=relative_frame)
        origin_coord = relative_coords.transform_to(self.coords)
        self.set_pos(photons, origin_coord)

        return photons


class SymbolFSource(AstroSource):
    '''Source shaped like the letter F.

    This source provides a non-symmetric source for testing purposes.

    Parameters
    ----------
    size : `astropy.units.quantity`
        angular size
    kwargs : see `Source`
        Other keyword arguments include ``flux``, ``energy`` and ``polarization``.
        See `Source` for details.
    '''
    def __init__(self, **kwargs):
        self.size = kwargs.pop('size', 1. * u.degree)
        super(SymbolFSource, self).__init__(**kwargs)

    @u.quantity_input()
    def generate_photons(self, exposuretime: u.s):
        photons = super(SymbolFSource, self).generate_photons(exposuretime)
        n = len(photons)
        elem = np.random.choice(3, size=n)

        ra = np.ones(n) * self.coords.icrs.ra
        dec = np.ones(n) * self.coords.icrs.dec
        size = self.size
        ra[elem == 0] += size * np.random.random(np.sum(elem == 0))
        ra[elem == 1] += size
        dec[elem == 1] += 0.5 * size * np.random.random(np.sum(elem == 1))
        ra[elem == 2] += 0.8 * size
        dec[elem == 2] += 0.3 * size * np.random.random(np.sum(elem == 2))

        self.set_pos(photons, SkyCoord(ra, dec, frame=self.coords))

        return photons
