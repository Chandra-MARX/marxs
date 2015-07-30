import numpy as np
from scipy.stats import expon
from astropy.table import Table, Column
from transforms3d.euler import euler2mat

from ..base import SimulationSequenceElement
from ..optics.polarization import polarization_vectors
from ..math.random import RandomArbitraryPdf


def poisson_process(rate):
    '''Return a function that generates Poisson distributed times with rate ``rate``.

    Parameters
    ----------
    rate : float
        Expectation value for the rate of events.

    Returns
    -------
    poisson_rate : function
        Function that generates Poisson distributed times with rate ``rate``.
    '''
    def poisson_rate(exposuretime):
        '''Generate Poisson distributed times.

        Parameters
        ----------
        exposuretime : float

        Returns
        -------
        times : `numpy.ndarray`
            Poisson distributed times.
        '''
        # Make 10 % more numbers then we expect to need, because it's random
        times = expon.rvs(scale=1./rate, size=exposuretime * rate * 1.1)
        # If we don't have enough numbers right now, add some more.
        while times.sum() < exposuretime:
            times = np.hstack([times, expon.rvs(scale=1/rate,
                                                size=(exposuretime - times.sum() * rate * 1.1))])
        times = np.cumsum(times)
        return times[times < exposuretime]
    return poisson_rate

class SourceSpecificationError(Exception):
    pass

class Source(SimulationSequenceElement):
    '''Base class for all photons sources.

    This class provides a very general implementation of photons sources. Typically,
    it is not used directly, but a more specialized subclass, such as `ConstantPointSource` for an
    astronomical source or `LabConstantPointSource` for a source at a finite distance.

    Most of the derived source support the same input argumets as `Source`, thus they are
    explained in detail here.

    Parameters
    ----------
    flux : number or callable
        This sets the total flux from a source in photons/s/effective area; the default value
        is 1 cts/s.
        Options are:

        - number: Constant (not Poisson distributed) flux.
        - callable: Function that takes a total exposure time as input and returns an array
          of photon emission times between 0 and the total exposure time.

    energy : number of callable or (2, N) `numpy.ndarray` or `numpy.recarracy` or `dict` or `astropy.table.Table`

        This input decides the energy of the emitted photons; the default value is 1 keV.
        Possible formats are:

        - number: Constant energy.
        - (2, N) `numpy.ndarray` or object with ``energy`` and ``flux`` columns (e.g. `dict` or
          `astropy.table.Table`), where "flux" here really is a short form for
          "flux density" and is given in the units of photons/s/keV.
          For a (2, N) array the first column is the energy, the
          second column is the flux density.
          Given this table, the code assumes a piecewise constant spectrum. The "energy"
          values contain the **upper** limit of each bin, the "flux" array the flux density
          in each bin. The first entry in the "flux" array is ignored, because the lower bound
          of this bin is undefined.
          The code draws an energy from this spectrum for every photon created.
        - A function or callable object: This option allows for full customization. The
          function must take an array of photon times as input and return an equal length
          array of photon energies in keV.

    polarization: contant or None, (2, N) `numpy.ndarray`, `dict`, `astropy.table.Table` or similar or callable.
        There are several different ways to set the polarization angle of the photons for a
        polarized source. In all cases, the angle is given in radian and is measured North
        through East. (We ignore the special case of a polarized source exactly on a pole.)
        The default value is ``None`` (unpolarized source).

        - ``None``:
          An unpolarized source. Every photons is assigned a random polarization.
        - number: Constant polarization angle for all photons.
        - (2, N) `numpy.ndarray` or object with ``angle`` and ``probability`` columns
          (e.g. `dict` or `astropy.table.Table`), where "probability" really means
          "probability density" here.
          The summed probability density will automatically be normalized to one.
          For a (2, N) array the first column is the angle, the second column is the
          probability *density*. Given this table, the code assumes a piecewise constant
          probability density. The "angle" values contain the **upper** limit of each bin,
          the "probability" array the probability density in this bin. The first entry in
          the "probability" array is ignored, because the lower bound
          of this bin is undefined. The code draws an energy from this spectrum for every
          photon created.
        - a callable (function or callable object): This option allows full customization.
          The function is called with two arrays (time and energy values) as input
          and must return an array of equal length that contains the polarization angles in
          radian.
    '''
    def __init__(self, **kwargs):
        self.energy = kwargs.pop('energy', 1.)
        self.flux = kwargs.pop('flux', 1.)
        self.polarization = kwargs.pop('polarization', None)

        super(Source, self).__init__(**kwargs)

    def generate_times(self, exposuretime):
        if callable(self.flux):
            return self.flux(exposuretime)
        elif np.isscalar(self.flux):
            return np.arange(0, exposuretime, 1./self.flux)
        else:
            raise SourceSpecificationError('`flux` must be a number or a callable.')

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
        return Table({'time': times, 'energy': energies, 'polangle': pol,
                      'probability': np.ones(n)})



class ConstantPointSource(Source):
    '''Astrophysical point source.

    Parameters
    ----------
    flux, energy, polarization : see `Source`
    coords : Tuple of 2 elements
        Ra and Dec in decimal degrees.
    '''
    def __init__(self, coords, **kwargs):
        self.coords = coords
        super(ConstantPointSource, self).__init__(**kwargs)

    def generate_photons(self, exposuretime):
        photons = super(ConstantPointSource, self).generate_photons(exposuretime)
        photons['ra'] = np.ones(len(photons)) * self.coords[0]
        photons['dec'] = np.ones(len(photons)) * self.coords[1]

        return photons

class SymbolFSource(Source):
    '''Source shaped like the letter F.

    This source provides a non-symmetric source for testing purposes.

    Parameters
    ----------
    flux, energy, polarization : see `Source`
    coords : tuple of 2 elements
        Ra and Dec in decimal degrees.
    size : float
        size scale in degrees
    '''
    def __init__(self, coords, size=1, **kwargs):
        self.coords = coords
        self.size = size
        super(ConstantPointSource, self).__init__(**kwargs)

    def generate_photons(self, exposuretime):
        photons = super(ConstantPointSource, self).generate_photons(exposuretime)
        n = len(photons)
        elem = np.random.choice(3, size=n)

        ra = np.empty(n)
        ra[:] = self.coords[0]
        dec = np.empty(n)
        dec[:] = self.coords[1]
        ra[elem == 0] += self.size * np.random.random(np.sum(elem == 0))
        ra[elem == 1] += self.size
        dec[elem == 1] += 0.5 * self.size * np.random.random(np.sum(elem == 1))
        ra[elem == 2] += 0.8 * self.size
        dec[elem == 2] += 0.3 * self.size * np.random.random(np.sum(elem == 2))

        photons['ra'] = ra
        photons['dec'] = dec

        return photons


class PointingModel(SimulationSequenceElement):
    '''A base model for all pointing models

    Conventions:

    - All angles (``ra``, ``dec``, and ``roll``) are given in decimal degrees.
    - x-axis points to sky aimpoint.
    - ``roll = 0`` means: z axis points North (measured N -> E).

    For :math:`\delta \pm 90^{\circ}` the :math:`\alpha` value is
    irrelevant for the pointing direction - any right ascension will
    lead to a pointing on the pole. A value for ``ra`` is still
    required, because it determines the orientation of the detector
    plane. Obviously, for pointing straight at the pole, the simple
    interpretation *z axis points north* is meaningless, but the
    combination of ``ra``, ``dec`` and ``roll`` still uniquely
    determines the position of the coordinate system.
    '''
    def add_dir(self, photons):
        linecoords = Column(name='dir', length=len(photons),
                            shape=(4,))
        photons.add_column(linecoords)
        # Leave everything unset, but chances are I will forget the 4th
        # component. Play safe here.
        photons['dir'][:, 3] = 0

    def process_photons(self, photons):
        # Could also loop over single photons if not implemented in
        # derived class.
        self.add_plucker(photons)
        raise NotImplementedError


class FixedPointing(PointingModel):
    '''Transform spacecraft to fixed sky system.

    This matrix transforms from the spacecraft system to a
    right-handed Cartesian system that is defined in the following
    way: the (x,y) plane is defined by the celestial equator, and
    the x-axis points to :math:`(\alpha, \delta) = (0,0)`.

    Parameters
    ----------
    coords : tuple of 2 elements
        Ra and Dec of telescope aimpoint in decimal degrees.
    roll : float
        ``roll = 0`` means: z axis points North (measured N -> E).

    Note
    ----
    For :math:`\delta \pm 90^{\circ}` the :math:`\alpha` value is
    irrelevant for the pointing direction - any right ascension will
    lead to a pointing on the pole. A value for ``ra`` is still
    required, because it determines the orientation of the detector
    plane. Obviously, for pointing straight at the pole, the simple
    interpretation *z axis points north* is meaningless, but the
    combination of ``ra``, ``dec`` and ``roll`` still uniquely
    determines the position of the coordinate system.

    Negative sign for dec and roll in the code, because these are defined
    opposite to the right hand rule.
    '''
    def __init__(self, **kwargs):
        self.coords = kwargs.pop('coords')
        self.ra = self.coords[0]
        self.dec = self.coords[1]
        if 'roll' in kwargs:
            self.roll = kwargs.pop('roll')
        else:
            self.roll = 0.

        super(FixedPointing, self).__init__(**kwargs)

        self.mat3d = euler2mat(np.deg2rad(self.ra),
                               np.deg2rad(-self.dec),
                               np.deg2rad(-self.roll), 'rzyx')

    def process_photons(self, photons):
        '''
        Parameters
        ----------
        photons : astropy.table.Table
        '''
        self.add_dir(photons)
        ra = np.deg2rad(photons['ra'])
        dec = np.deg2rad(photons['dec'])
        # Minus sign here because photons start at +inf and move towards origin
        photons['dir'][:, 0] = - np.cos(dec) * np.cos(ra)
        photons['dir'][:, 1] = - np.cos(dec) * np.sin(ra)
        photons['dir'][:, 2] = - np.sin(dec)
        photons['dir'][:, 3] = 0
        photons['dir'][:, :3] = np.dot(np.linalg.inv(self.mat3d), photons['dir'][:, :3].T).T
        photons['polarization'] = np.ones_like(ra)
        return photons
