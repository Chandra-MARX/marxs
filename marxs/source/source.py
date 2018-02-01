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


def poisson_process(rate):
    '''Return a function that generates Poisson distributed times with rate ``rate``.

    Parameters
    ----------
    rate : float
        Expectation value for the total rate of events in photons / cm**2 / s.

    Returns
    -------
    poisson_rate : function
        Function that generates Poisson distributed times with rate ``rate``.
    '''
    if not np.isscalar(rate):
        raise ValueError('"rate" must be scalar.')

    def poisson_rate(exposuretime, geomarea):
        '''Generate Poisson distributed times.

        Parameters
        ----------
        exposuretime : float
            Exposure time in sec.
        geomarea : `astropy.unit.Quantity`
            Geometric opening area of telescope

        Returns
        -------
        times : `numpy.ndarray`
            Poisson distributed times.
        '''
        fullrate = rate * geomarea.to(u.cm**2).value
        # Make 10 % more numbers then we expect to need, because it's random
        times = expon.rvs(scale=1./fullrate, size=int(exposuretime * fullrate * 1.1))
        # If we don't have enough numbers right now, add some more.
        while times.sum() < exposuretime:
            times = np.hstack([times, expon.rvs(scale=1/fullrate,
                                                size=int(exposuretime - times.sum() * fullrate * 1.1))])
        times = np.cumsum(times)
        return times[times < exposuretime]
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
    flux : number or callable This sets the total flux from a source in
        photons/s/cm^2; the default value is 1 cts/s.  Options are:

        - number: Constant (not Poisson distributed) flux.
        - callable: Function that takes a total exposure time as input and
          returns an array
          of photon emission times between 0 and the total exposure time.

    energy : number of callable or (2, N) `numpy.ndarray` or `numpy.recarray` or `dict <dict>` or `astropy.table.Table`

        This input decides the energy of the emitted photons;
        the default value is 1 keV.
        Possible formats are:

        - number: Constant energy.
        - (2, N) `numpy.ndarray` or object with columns "energy" and "flux"
          (e.g. `dict <dict>` or `astropy.table.Table`),
          where "flux" here really is a short form for
          "flux density" and is given in the units of photons/s/keV.
          For a (2, N) array the first column is the energy, the
          second column is the flux density.
          Given this table, the code assumes a piecewise flat spectrum.
          The "energy" values contain the **upper** limit of each bin,
          the "flux" array the flux density in each bin.
          The first entry in the "flux" array is ignored, because the lower
          bound of this bin is undefined.
          The code draws an energy from this spectrum for every photon created.
        - A function or callable object: This option allows for full
          customization. The function must take an array of photon times as
          input and return an equal length array of photon energies in keV.

    polarization: contant or ``None``, (2, N) `numpy.ndarray`, `dict <dict>`, `astropy.table.Table` or similar or callable.
        There are several different ways to set the polarization angle of the
        photons for a polarized source. In all cases, the angle is measured
        North through East. (We ignore the special case of a polarized source
        exactly on a pole.)
        The default value is ``None`` (unpolarized source).

        - ``None``: An unpolarized source. Every photons is assigned a random
          polarization.
        - number: Constant polarization angle for all photons
          (in degrees).
        - (2, N) `numpy.ndarray` or object with columns
          "angle" and "probability" (e.g. `dict <dict>` or
          `astropy.table.Table`), where "probability" really means "probability
          density".  The summed probability density will automatically be
          normalized to one.  For a (2, N) array the first column is the angle,
          the second column is the probability *density*. Given this table, the
          code assumes a piecewise constant probability density. The "angle"
          values contain the **upper** limit of each bin, the "probability"
          array the probability density in this bin. The first entry in the
          "probability" array is ignored, because the lower bound of this bin
          is undefined.
        - a callable (function or callable object): This
          option allows full customization.  The function is called with two
          arrays (time and energy values) as input and must return an array of
          equal length that contains the polarization angles in degrees.

    geomarea : `astropu.units.Quantity` Geometric opening area of
          telescope. Default is :math:`1 cm^2`.

    '''
    def __init__(self, **kwargs):
        self.energy = kwargs.pop('energy', 1.)
        self.flux = kwargs.pop('flux', 1.)
        self.polarization = kwargs.pop('polarization', None)
        self.geomarea = kwargs.pop('geomarea', 1. * u.cm**2)

        super(Source, self).__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.generate_photons(*args, **kwargs)

    def generate_times(self, exposuretime):
        if callable(self.flux):
            return self.flux(exposuretime, self.geomarea)
        elif np.isscalar(self.flux):
            return np.arange(0, exposuretime, 1./(self.flux * self.geomarea.to(u.cm**2).value))
        else:
            raise SourceSpecificationError('`flux` must be a quantity or a callable.')

    def generate_energies(self, t):
        n = len(t)
        # function
        if callable(self.energy):
            en = self.energy(t)
            if len(en) != n:
                raise SourceSpecificationError('`energy` has to return an array of same size as input time array.')
            else:
                return en
        # constant energy
        elif np.isscalar(self.energy):
            return np.ones(n) * self.energy
        # 2 * n numpy array
        elif hasattr(self.energy, 'shape') and (self.energy.shape[0] == 2):
            rand = RandomArbitraryPdf(self.energy[0, :], self.energy[1, :])
            return rand(n)
        # np.recarray or astropy.table.Table
        elif hasattr(self.energy, '__getitem__'):
            rand = RandomArbitraryPdf(self.energy['energy'], self.energy['flux'])
            return rand(n)
        # anything else
        else:
            raise SourceSpecificationError('`energy` must be number, function, 2*n array or have fields "energy" and "flux".')


    def generate_polarization(self, times, energies):
        n = len(times)
        # function
        if callable(self.polarization):
            pol = self.polarization(times, energies)
            if len(pol) != n:
                raise SourceSpecificationError('`polarization` has to return an array of same size as input time and energy arrays.')
            else:
                return pol
        elif np.isscalar(self.polarization):
            return np.ones(n) * self.polarization
        # 2 * n numpy array
        elif hasattr(self.polarization, 'shape') and (self.polarization.shape[0] == 2):
            rand = RandomArbitraryPdf(self.polarization[0, :], self.polarization[1, :])
            return rand(n)
        # np.recarray or astropy.table.Table
        elif hasattr(self.polarization, '__getitem__'):
            rand = RandomArbitraryPdf(self.polarization['angle'], self.polarization['probability'])
            return rand(n)
        elif self.polarization is None:
            return np.random.uniform(0, 2 * np.pi, n)
        else:
            raise SourceSpecificationError('`polarization` must be number (angle), callable, None (unpolarized), 2.n array or have fields "angle" (in rad) and "probability".')

    def generate_photon(self):
        raise NotImplementedError

    def generate_photons(self, exposuretime):
        '''Central function to generate photons.

        Calling this function generates a a photon table according to the `flux`, `energy`,
        and `polarization` of this source. The number of photons depends on the total
        exposure time, which is a parameter of this function. Depending on the setting for
        `flux` the photons could be distributed equally over the interval 0..exposuretime
        or follow some other distribution.

        Parameters
        ----------
        exposuretime : float
            Total exposure time in seconds.

        Returns
        -------
        photons : `astropy.table.Table`
            Table with photon properties.
        '''
        times = self.generate_times(exposuretime)
        energies = self.generate_energies(times)
        pol = self.generate_polarization(times, energies)
        n = len(times)
        photons = Table([times, energies, pol, np.ones(n)],
                        names=['time', 'energy', 'polangle', 'probability'])
        photons.meta['EXTNAME'] = 'EVENTS'
        photons.meta['EXPOSURE'] = (exposuretime, 'total exposure time [s]')

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

        This function write Ra, Dec to a table. It is defined here to make the way
        `astropy.coordinates.SkyCoord` objects are stored more uniform.
        Currently, mixin columns in tables have some disadvantages, e.g. they
        cause errors on writing and on stacking. Thus, we store the coordinates
        as plain numbers. Since that format is not unique (e.g. units could be deg or rad),
        system could be ICRS, FK4, FK5 or other this conversion is done here for all
        astrononimcal sources.
        This also makes it easier to change that design in the future.

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

    def generate_photons(self, exposuretime):
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

    def generate_photons(self, exposuretime):
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
        kwargs['radial_distribution'] = lambda n: np.arcsin(np.cos(self.func_par[1]) +
                          np.random.rand(n) * (np.cos(self.func_par[0]) - np.cos(self.func_par[1])))
        super(SphericalDiskSource, self).__init__(**kwargs)

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

class GaussSource(RadialDistributionSource):
    '''Astrophysical source with a Gaussian brightness profile.

    This source uses a small angle approximation which is valid for radii less than
    a few degrees.

    Parameters
    ----------
    sigma : `astropy.coordinates.Angle`
        Gaussian sigma setting the width of the distribution
    '''
    def __init__(self, **kwargs):
        kwargs['func_par'] = kwargs.pop('sigma')
        # Note: rand is in interavall [0..1[, so 1-rand is the same except for edges
        kwargs['radial_distribution'] = lambda n: self.func_par * np.sqrt(np.log(np.random.rand(n)))
        super(GaussSource, self).__init__(**kwargs)

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

    def generate_photons(self, exposuretime):
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
