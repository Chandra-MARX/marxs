from warnings import warn

import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.utils import normalized_vector

from astropy.table import Column
from ..base import SimulationSequenceElement
from ..math.pluecker import h2e, e2h
from ..math.rotations import axangle2mat
from ..utils import SimulationSetupWarning


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
        self.add_dir(photons)
        return photons



class FixedPointing(PointingModel):
    r'''Transform spacecraft to fixed sky system.

    This matrix transforms from the spacecraft system to a
    right-handed Cartesian system that is defined in the following
    way: the (x,y) plane is defined by the celestial equator, and
    the x-axis points to :math:`(\alpha, \delta) = (0,0)`.

    Parameters
    ----------
    coords : tuple of 2 elements
        Ra and Dec of telescope aimpoint in decimal degrees.
    roll : float
        ``roll = 0`` means: z axis points North (measured N -> E). Angle in degrees.
    geomarea_reference : np.array
        Homogeneous vector that is normal (with a direction pointing from the
        the source towards the instrument) to all apertures of the telescope.
        Sources that do not shine directly onto the telescope aperture but
        hit it at an angle, will see a smaller projected geometric area.
        This is taken into account by reducing the probability of off-axies photons
        accordingly, and thus this objects needs to know the normal (the optical
        axis) of the aperture. Default is the x-axis, with sources at
        :math:`+\infty` and pointing inward, i.e. ``[-1, 0, 0, 0]``.

    Notes
    -----
    For :math:`\delta \pm 90^{\circ}` the :math:`\alpha` value is
    irrelevant for the pointing direction - any right ascension will
    lead to a pointing on the pole. A value for ``ra`` is still
    required, because it determines the orientation of the detector
    plane. Obviously, for pointing straight at the pole, the simple
    interpretation *z axis points north* is meaningless, but the
    combination of ``ra``, ``dec`` and ``roll`` still uniquely
    determines the position of the coordinate system.
    '''
    def __init__(self, **kwargs):
        self.coords = kwargs.pop('coords')
        self.ra = self.coords[0]
        self.dec = self.coords[1]
        self.roll = kwargs.pop('roll', 0.)
        self.geomarea_reference = normalized_vector(kwargs.pop('geomarea_reference',
                                                               np.array([-1, 0, 0, 0])))

        super(FixedPointing, self).__init__(**kwargs)
        '''
        Negative sign for dec and roll in the code, because these are defined
        opposite to the right hand rule.
        '''
        self.mat3d = euler2mat(np.deg2rad(self.ra),
                               np.deg2rad(-self.dec),
                               np.deg2rad(-self.roll), 'rzyx')

    def photons_dir(self, ra, dec, time):
        '''Calculate direction on photons in homogeneous coordinates.

        Parameters
        ----------
        ra : np.array
            RA for each photon in rad
        dec : np.array
            DEC or each photon in rad
        time : np.array
            Time for each photons in sec

        Returns
        -------
        photons_dir : np.array of shape (n, 4)
            Homogeneous direction vector for each photon
        '''
        # Minus sign here because photons start at +inf and move towards origin
        photons_dir = np.zeros((len(ra), 4))
        photons_dir[:, 0] = - np.cos(dec) * np.cos(ra)
        photons_dir[:, 1] = - np.cos(dec) * np.sin(ra)
        photons_dir[:, 2] = - np.sin(dec)
        photons_dir[:, :3] = np.dot(self.mat3d.T, photons_dir[:, :3].T).T

        return photons_dir

    def photons_pol(self, ra, dec, polangle, time):
        return np.ones_like(ra)

    def process_photons(self, photons):
        '''
        Parameters
        ----------
        photons : astropy.table.Table
        '''
        photons = super(FixedPointing, self).process_photons(photons)
        ra = np.deg2rad(photons['ra'].data)
        dec = np.deg2rad(photons['dec'].data)
        photons['dir'] = self.photons_dir(ra, dec, photons['time'].data)
        projected_area = np.dot(photons['dir'].data, self.geomarea_reference)
        # Photons coming in "through the back" would have negative probabilities.
        # Unlikely to ever come up, but just in case we clip to 0.
        photons['probability'] *= np.clip(projected_area, 0, 1.)
        photons['polarization'] = self.photons_pol(ra, dec, photons['time'].data, photons['polangle'].data)
        photons.meta['RA_PNT'] = (self.ra, '[deg] Pointing RA')
        photons.meta['DEC_PNT'] = (self.dec, '[deg] Pointing Dec')
        photons.meta['ROLL_PNT'] = (self.roll, '[deg] Pointing Roll')
        photons.meta['RA_NOM'] = (self.ra, '[deg] Nominal Pointing RA')
        photons.meta['DEC_NOM'] = (self.dec, '[deg] Nominal Pointing Dec')
        photons.meta['ROLL_NOM'] = (self.roll, '[deg] Nominal Pointing Roll')

        return photons


class JitterPointing(FixedPointing):
    '''Transform spacecraft to fixed sky system.

    This extends `marxs.sourcs.FixedPointing` by adding a random jitter coordinate.
    In this simple implementation the jitter angles applied to two consecutive
    photons are entirely uncorrelated, even if these two photons arrive at the
    same time.
    This class makes the assumption that jitter is small (no change in the projected
    geometric area of the aperture due to jitter).

    Parameters
    ----------
    jitter : float
        Gaussian sigma of jitter angle in radian
    '''
    def __init__(self, **kwargs):
        self.jitter = kwargs.pop('jitter')
        if self.jitter > 1e-4:
            warn('Jitter is {0} which seems large [jitter is expected in radian, not arcsec].'.format(self.jitter), SimulationSetupWarning)
        super(JitterPointing, self).__init__(**kwargs)

    def photons_dir(self, ra, dec, time):
        photons_dir = super(JitterPointing, self).photons_dir(ra, dec, time)
        # Get random jitter direction
        randang = np.random.rand(len(ra)) * 2. * np.pi
        ax = np.vstack([np.zeros_like(ra), np.sin(randang), np.cos(randang)]).T
        jitterang = np.random.normal(scale=self.jitter, size=len(ra))
        jitterrot = axangle2mat(ax, jitterang)
        photons_dir = np.einsum('...ij,...i->...j', jitterrot, h2e(photons_dir))
        return e2h(photons_dir, 0)
