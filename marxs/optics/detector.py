from __future__ import division
import warnings

import numpy as np
from np.linalg import norm
from transforms3d.affines import decompose44

from .base import FlatOpticalElement, OpticalElement

class PixelSizeWarning(Warning):
    pass

warnings.filterwarnings("always", ".*", PixelSizeWarning)

class FlatDetector(FlatOpticalElement):
    '''Flat detector with square pixels

    Processing the a photon with this detector adds four columns to the photon
    properties:

    - ``det_x``, ``det_y``: Event position on the detector in mm
    - ``detpix_x``, ``detpix_y``: Event position in detector coordinates, where
      (0, 0) is on the corner of the chip.

    The (x,y) coordinate system on the chip is such that it falls on the (y,z) plane
    of the global x,y,z coordinate system (it would be more logical to call the chip
    system (y,z), but traditionally that is not how chip coordinates are named). The
    pixel in the corner has coordinates (0, 0) in the pixel center.

    Parameters
    ----------
    pixsize : float
        size of pixels in mm

    kwargs :
       see `args for optical elements`
    '''

    loc_coos_name = ['det_x', 'det_y']
    '''name for output columns that contain the interaction point in local coordinates.'''

    detpix_name = ['detpix_x', 'detpix_y']
    '''name for output columns that contain this pixel number.'''

    display = {'color': (1.0, 1.0, 0.)}

    def __init__(self, pixsize=1, **kwargs):
        self.pixsize = pixsize
        super(FlatDetector, self).__init__(**kwargs)
        t, r, zoom, s = decompose44(self.pos4d)
        self.npix = [0, 0]
        self.centerpix = [0, 0]
        for i in (0, 1):
            z  = zoom[i + 1]
            self.npix[i] = int(np.round(2. * z / self.pixsize))
            if np.abs(2. * z / self.pixsize - self.npix[i]) > 1e-2:
                warnings.warn('Detector size is not an integer multiple of pixel size in direction {0}. It will be rounded.'.format('xy'[i]), PixelSizeWarning)
            self.centerpix[i] = (self.npix[i] - 1) / 2

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        detx = intercoos[intersect, 0] / self.pixsize + self.centerpix[0]
        dety = intercoos[intersect, 1] / self.pixsize + self.centerpix[1]
        return {self.detpix_name[0]: detx, self.detpix_name[1]: dety}

class CircularDetector(OpticalElement):
    '''A detector shaped like a ring or tube.

    This detector is shaped like a tube. The form is a circle in the xy plane and
    and flat along the z-direction.
    While most CCDs are flat in practice, the `CircularDetector` simulates a setup
    that can follow the Rowland circle geometry exactly which is useful, e.g. to study
    the resolution of a spectrograph without worrying about the details of the detector
    geometry.

    Parameters
    ----------

    '''
    loc_coos_name = ['det_phi', 'det_y']

    detpix_name = ['detpix_x', 'detpix_y']
    '''name for output columns that contain this pixel number.'''

    def __init__(self, pixsize=1, **kwargs):
        self.pixsize = pixsize
        self.r = kwargs.pop('r')
        self.phi_offset = kwargs.pop('phi_offset', 0.)
        self._inwards = kwargs.pop('inwards', True)
        super(CircularDetector, self).__init__(**kwargs)

    @property
    def inwardsoutwards(self):
        if self._inwards:
            return 1.
        else:
            return -1.

    def intersect(self, dir, pos, transform=True):
        '''Calculate the intersection point between a ray and the element

        Parameters
        ----------
        dir : `numpy.ndarray` of shape (N, 4)
            homogeneous coordinates of the direction of the ray
        pos : `numpy.ndarray` of shape (N, 4)
            homogeneous coordinates of a point on the ray
        transform : bool
            If ``True``, input is in global coordinates and needs to be transformed
            here for the calculations; if ``False`` input is in local coordinates.

        Returns
        -------
        intersect :  boolean array of length N
            ``True`` if an intersection point is found.
        interpos : `numpy.ndarray` of shape (N, 4)
            homogeneous coordinates of the intersection point. Values are set
            to ``np.nan`` is no intersecton point is found.
        interpos_local : `numpy.ndarray` of shape (N, 2)
            phi, z coordiantes (in the local frame) for one of the intersection points.
            If both intersection points are required, reset ``self.inner`` and call this
            function again.
        '''
        if transform:
            invpos4d = np.linalg.inv(self.pos4d)
            dir = np.dot(invpos4d, dir.T).T
            pos = np.dot(invpos4d, pos.T).T

        xyz = h2e(pos)

        # Solve quadratic equation in steps. a12 = (-xr +- sqrt(xr - r**2(x**2 - R**2)))
        xy = xyz[:, :2]
        r = dir[:, :2]
        underroot = np.dot(xy, r) - np.dot(r, r) * (np.dot(xy, xy) - self.r**2)
        intersect = (underroot >= 0)
        i = intersect  # just a shorthand because it's used so much below

        interpos_local = np.ones((pos.shape[0], 2))
        interpos_local[:] = np.nan
        interpos = np.ones_like(pos)
        interpos[:] = np.nan

        if intersect.sum() > 0:
            b = np.dot(xy[i], r[i])
            denom = np.dor(r[i], r[i])
            a1 = (- b + np.sqrt(underroot[i])) / denom
            a2 = (- b - np.sqrt(underroot[i])) / denom
            interpos = np.ones((4, intersect.sum()))
            x1 = xy[i] + a1 * r[i]
            apick = np.where(self.inwardsoutwards * np.dot(x1, r) >=0, a1, a2)
            xy_p = xy + apick * r
            interpos_local[intersect, 0] = np.arctan2(xy_p[:, 1], xy_p[:, 2]) + self.phi_offset
            # Those look like they hit in the xy plane.
            # Still possible to miss if z axis is too large.
            # Calculate z-coordiante at intersection
            interpos_local[intersect, 1] = xyz[:, 2] + apick * dir[2, :]
            interpos[intersect, :2] = xy_p
            interpos[intersect, 2] = interpos_local[intersect, 1]
            interpos[intersect, 3] = 1
            # set those elements on intersect that miss in z to False
            trans, rot, zoom, shear = decompose44(self.pos4d)
            intersect[intersect.nonzero()[np.abs(z_p) > zoom[2]]] = False
            # Now reset everything to nan that is not intersect
            interpos_local[~intersect, :] = np.nan
            interpos[~intersect, :] = np.nan

            interpos = np.dot(self.pos4d, interpos.T).T

        return intersect, interpos, interpos_local

    def process_photons(self, photons):
        intersect, interpos, interpos_local = self.intersect(photons['dir'], photons['pos'])

        photons['pos'][intersect, :] = interpos[intersect, :]
        self.add_output_cols(photons, self.loc_coos_name + self.detpix_name)
        # Add ID number to ID col, if requested
        if self.id_col is not None:
            photons[self.id_col][intersect] = self.id_num
        # Set position in different coordinate systems
        photons['pos'][intersect] = interpos[intersect]
        photons[self.loc_coos_name[0]][intersect] = interpos_local[intersect, 0]
        photons[self.loc_coos_name[1]][intersect] = interpos_local[intersect, 1]
        photons[self.detpix_name[0]][intersect] = intercoos[intersect, 0] * self.r / self.pixsize
        photons[self.detpix_name[0]][intersect] = intercoos[intersect, 0] * self.r / self.pixsize

        return photons
