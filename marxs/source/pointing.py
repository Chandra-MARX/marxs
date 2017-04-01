# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from astropy.table import Column
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord

from ..base import SimulationSequenceElement
from ..math.rotations import axangle2mat
from ..math.utils import norm_vector, h2e, e2h


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
    reference_transform : np.array of shape (4, 4)
        By default, photons from an on-axis source come in parallel to the x-axis
        of the coordinate system. Their direction points from x=+inf inwards.
        If the simulation uses a different coordinate system (e.g. the optical
        axis is along the z-axis) set ``reference_transform`` to a matrix that
        performs the conversion.

        The optical axis of the telescope is the normal to the surface of its
        entrance aperture. The pointing needs to know this to determine
        the correct direction of the photons.
        Also, sources that do not shine directly onto the telescope aperture but
        hit it at an angle, will see a smaller projected geometric area.
        This is taken into account by reducing the probability of off-axies photons
        accordingly, and thus this object needs to know the orientation (the
        direction f the optical axis and rotation) of the aperture.

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

        self.reference_transform = kwargs.pop('reference_transform', np.eye(4))

        super(FixedPointing, self).__init__(**kwargs)

    @property
    def offset_coos(self):
        '''Return `~astropy.coordinates.SkyOffsetFrame`'''
        return self.coords.skyoffset_frame(rotation=self.roll)

    def photons_dir(self, coos, time):
        '''Calculate direction of photons in homogeneous coordinates.

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
        # Minus sign here because photons start at +inf and move towards origin
        photonsdir = norm_vector(-photondir.cartesian.xyz.T)
        return np.einsum('...ij,...j->...i', self.reference_transform, e2h(photonsdir, 0))

    def photons_pol(self, photonsdir, polangle, time):
        '''Calculate a polarization vector for linearly polarized light.

        The current definition cannot handle photons coming exactly from either
        the North pole or the South Pole of the sphere, because the polangle
        definition "North through east" is not well-defined in these positions.

        Parameters
        ----------
        photonsdir : np.array of shape (n, 4)
            Direction of photons
        polangle : np.array
            Polarization angle in degree measured N through E.
        time : np.array
            Time for each photons in sec
        '''
        polangle = np.deg2rad(polangle)
        north = SkyCoord(0., 90., unit='deg', frame=self.coords)
        northdir = e2h(north.transform_to(self.offset_coos).cartesian.xyz.T, 0)
        northdir = np.dot(self.reference_transform, northdir)
        n_inskyplane = norm_vector(northdir - photonsdir * np.dot(northdir, photonsdir.T)[:, None])
        e_inskyplane = e2h(np.cross(photonsdir[:, :3], n_inskyplane[:, :3]), 0)
        return  np.cos(polangle)[:, None] * n_inskyplane + np.sin(polangle)[:, None] * e_inskyplane

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
        photons['polarization'] = self.photons_pol(photons['dir'].data,
                                                   photons['polangle'].data,
                                                   photons['time'].data)
        photons.meta['RA_PNT'] = (self.coords.ra.degree, '[deg] Pointing RA')
        photons.meta['DEC_PNT'] = (self.coords.dec.degree, '[deg] Pointing Dec')
        photons.meta['ROLL_PNT'] = (self.roll.to(u.degree).value,
                                    '[deg] Pointing Roll')
        photons.meta['RA_NOM'] = (self.coords.ra.degree, '[deg] Nominal Pointing RA')
        photons.meta['DEC_NOM'] = (self.coords.dec.degree, '[deg] Nominal Pointing Dec')
        photons.meta['ROLL_NOM'] = (self.roll.to(u.degree).value,
                                    '[deg] Nominal Pointing Roll')

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

    def process_photons(self, photons):
        photons = super(JitterPointing, self).process_photons(photons)
        # Get random jitter direction
        n = len(photons)
        randang = np.random.rand(n) * 2. * np.pi
        ax = np.vstack([np.zeros(n), np.sin(randang), np.cos(randang)]).T
        jitterang = np.random.normal(scale=self.jitter.to(u.radian).value, size=n)
        jitterrot = axangle2mat(ax, jitterang)
        photons['dir'] = e2h(np.einsum('...ij,...i->...j', jitterrot,
                                       h2e(photons['dir'])), 0)
        photons['polarization'] = e2h(np.einsum('...ij,...i->...j', jitterrot,
                                                h2e(photons['polarization'])), 0)
        return photons
