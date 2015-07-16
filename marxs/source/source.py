import numpy as np
from astropy.table import Table, Column
from transforms3d.euler import euler2mat

from ..base import SimulationSequenceElement
from ..optics.polarization import polarization_vectors

'''TO-DO: make proper object hirachy, always call super on init. Pass args through, e.g. for name.'''

class Source(SimulationSequenceElement):
    '''Make this ABC once I have worked out the interface'''
    def generate_photon(self):
        pass

    def generate_photons(self):
        # Similar to optics this could be given if generate_photons is given!
        pass
    
    def random_polarization(self, n, dir_array):
        # randomly choose polarization (temporary)
        angles = np.random.uniform(0, 2 * np.pi, n)
        return polarization_vectors(dir_array, angles)


class ConstantPointSource(Source):
    '''Simplest possible source for testing purposes:

    - constant energy
    - constant flux (I mean constant, not Poisson distributed - this is not
      an astrophysical source, it's for code testing.)
    '''
    def __init__(self, coords, flux, energy, polarization=np.nan, effective_area=1):
        self.coords = coords
        self.flux = flux
        self.effective_area = effective_area
        self.rate = flux * effective_area
        self.energy = energy
        self.polarization = polarization

    def generate_photons(self, t):
        n = t  * self.rate
        out = np.empty((6, n))
        out[0, :] = np.arange(0, t, 1. / self.rate)
        for i, val in enumerate([self.coords[0], self.coords[1], self.energy, self.polarization]):
            out[i + 1, :] = val
        out[5, :] = 1.
        return Table(out.T, names = ('time', 'ra', 'dec', 'energy', 'polarization', 'probability'))

class SymbolFSource(Source):
    '''Source shaped like the letter F.

    This source provies a non-symmetric source for testing purposes.

    Parameters
    ----------
    size : float
        size scale in degrees
    '''
    def __init__(self, coords, flux, energy, polarization=np.nan, effective_area=1, size=1):
        self.coords = coords
        self.flux = flux
        self.effective_area = effective_area
        self.rate = flux * effective_area
        self.energy = energy
        self.polarization = polarization
        self.size = size

    def generate_photons(self, t):
        n = t  * self.rate
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


        out = np.empty((6, n))
        out[0, :] = np.arange(0, t, 1. / self.rate)
        out[1,:] = ra
        out[2,:] = dec
        out[3, :] = self.energy
        out[4,:] = self.polarization
        out[5, :] = 1.
        return Table(out.T, names = ('time', 'ra', 'dec', 'energy', 'polarization', 'probability'))



class PointingModel(SimulationSequenceElement):
    '''A base model for all pointing models

    Conventions:

    - All angles (``ra``, ``dec``, and ``roll`` are given in decimal degrees.
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
    def __init__(self, **kwargs):
        self.coords = kwargs.pop('coords')
        self.ra = self.coords[0]
        self.dec = self.coords[1]
        if 'roll' in kwargs:
            self.roll = kwargs.pop('roll')
        else:
            self.roll = 0.

        super(FixedPointing, self).__init__(**kwargs)

        '''Transform spacecraft to fixed sky system.

        This matrix transforms from the spacecraft system to a
        right-handed Cartesian system that is defined in the following
        way: the (x,y) plane is defined by the celestial equator, and
        the x-axis points to :math:`(\alpha, \delta) = (0,0)`.

        Implementation notes
        --------------------
        Negative sign for dec and roll, because these are defined
        opposite to the right hand rule.

        Note: I hope I figured it out right. If not, I can always bail
        and use astropy.coordinates for this.
        '''
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
        return photons
