from warnings import warn

import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.utils import normalized_vector

from astropy.table import Column
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
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
    coords : `astropy.coordinates.SkySoord` (preferred)
        Position of the source on the sky. If ``coords`` is not a
        `~astropy.coordinates.SkyCoord` object itself, it is used to
        initialize such an object. See `~astropy.coordinates.SkyCoord`
        for a description of allowed input values.
    roll : `~astropy.units.quantity.Quantity`
        ``roll = 0`` means: z axis points North (measured N -> E).
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
        coords = kwargs.pop('coords')
        if isinstance(coords, coord.SkyCoord):
            self.coords = coords
        else:
            self.coords = coord.SkyCoord(coords)

        if not self.coords.isscalar:
            raise ValueError("Coordinate must be scalar, not array.")
        self.roll = kwargs.pop('roll', 0. * u.rad)

        self.geomarea_reference = normalized_vector(kwargs.pop('geomarea_reference',
                                                               np.array([-1, 0, 0, 0])))

        super(FixedPointing, self).__init__(**kwargs)

    @property
    def offset_coos(self):
        '''Return `~astropy.coordinates.SkyOffsetFrame`'''
        return self.coords.skyoffset_frame(rotation=self.roll)

    def photons_dir(self, coos, time):
        '''Calculate direction on photons in homogeneous coordinates.

        Parameters
        ----------
        coos : `astropy.coordiantes.SkyCoord`
            Origin of each photon on the sky
        time : np.array
            Time for each photons in sec

        Returns
        -------
        photons_dir : np.array of shape (n, 4)
            Homogeneous direction vector for each photon
        '''
        photondir = coos.transform_to(self.offset_coos)
        assert np.allclose(photondir.distance.value, 1.)
        # Minus sign here because photons start at +inf and move towards origin
        return - e2h(photondir.cartesian.xyz.T, 0)

    def photons_pol(self, coos, polangle, time):
        '''
        Parameters
        ----------
        coos : `astropy.coordiantes.SkyCoord`
            Origin of each photon on the sky
        polangle :
        time : np.array
            Time for each photons in sec
        '''
        return np.ones_like(time)

    def process_photons(self, photons):
        '''
        Parameters
        ----------
        photons : `astropy.table.Table`
        '''
        photons = super(FixedPointing, self).process_photons(photons)
        photons['dir'] = self.photons_dir(SkyCoord(photons['ra'], photons['dec'],
                                                   unit='deg'),
                                          photons['time'].data)
        projected_area = np.dot(photons['dir'].data, self.geomarea_reference)
        # Photons coming in "through the back" would have negative probabilities.
        # Unlikely to ever come up, but just in case we clip to 0.
        photons['probability'] *= np.clip(projected_area, 0, 1.)
        photons['polarization'] = self.photons_pol(SkyCoord(photons['ra'], photons['dec'],
                                                            unit='deg'),
                                                   photons['time'].data, photons['polangle'].data)
        photons.meta['RA_PNT'] = (self.coords.ra.degree, '[deg] Pointing RA')
        photons.meta['DEC_PNT'] = (self.coords.dec.degree, '[deg] Pointing Dec')
        photons.meta['ROLL_PNT'] = (self.roll.to(u.degree), '[deg] Pointing Roll')
        photons.meta['RA_NOM'] = (self.coords.ra.degree, '[deg] Nominal Pointing RA')
        photons.meta['DEC_NOM'] = (self.coords.dec.degree, '[deg] Nominal Pointing Dec')
        photons.meta['ROLL_NOM'] = (self.roll.to(u.degree), '[deg] Nominal Pointing Roll')

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
    jitter : `~astropy.units.quantity.Quantity`
        Gaussian sigma of jitter angle
    '''
    def __init__(self, **kwargs):
        self.jitter = kwargs.pop('jitter')
        super(JitterPointing, self).__init__(**kwargs)

    def photons_dir(self, *args):
        photons_dir = super(JitterPointing, self).photons_dir(*args)
        # Get random jitter direction
        n = len(photons_dir)
        randang = np.random.rand(n) * 2. * np.pi
        ax = np.vstack([np.zeros(n), np.sin(randang), np.cos(randang)]).T
        jitterang = np.random.normal(scale=self.jitter.to(u.radian), size=n)
        jitterrot = axangle2mat(ax, jitterang)
        photons_dir = np.einsum('...ij,...i->...j', jitterrot, h2e(photons_dir))
        return e2h(photons_dir, 0)
