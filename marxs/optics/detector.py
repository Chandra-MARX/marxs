# Licensed under GPL version 3 - see LICENSE.rst
from __future__ import division
import warnings

import numpy as np
from transforms3d.affines import decompose44, compose

from .base import FlatOpticalElement, OpticalElement
from ..math.utils import h2e
from ..visualization.utils import get_color
from ..utils import SimulationSetupWarning


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

    display = {'color': (1.0, 1.0, 0.),
               'shape': 'box'}

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
                warnings.warn('Detector size is not an integer multiple of pixel size in direction {0}. It will be rounded.'.format('xy'[i]), SimulationSetupWarning)
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
    position, orientation, zoom, pos4d : see description of `pos4d`
        The radius of the tube is given by the ``zoom`` keyword, see `pos4d`.
        Use ``zoom[0] == zoom[1]`` to make a circular tube. ``zoom[0] != zoom[1]`` gives
        an elliptical profile. ``zoom[2]`` sets the extension in the z direction.
    pixsize : float
        size of pixels in mm
    phi_offset : float
        This defines the center (pixel = 0) in radian. In other words:
        The output ``det_phi`` column has its zero position at ``phi_offset`` and
        values for ``phi`` are in the range [-pi, pi].
    '''
    loc_coos_name = ['det_phi', 'det_y']

    detpix_name = ['detpix_x', 'detpix_y']
    '''name for output columns that contain this pixel number.'''

    display = {'color': (1.0, 1.0, 0.),
               'opacity': 0.7,
               'shape': 'surface',
               'coo1': np.linspace(0, 2 * np.pi, 50),
               'coo2': [-1, 1]}

    def __init__(self, pixsize=1, **kwargs):
        self.pixsize = pixsize
        self.phi_offset = kwargs.pop('phi_offset', 0.)
        self.inwards = kwargs.pop('inside', True)
        super(CircularDetector, self).__init__(**kwargs)

    @classmethod
    def from_rowland(cls, rowland, width):
        '''Generate a `CircularDetector` from a `RowlandTorus`.

        Parameters
        ----------
        rowland : `~marxs.design.RowlandTorus`
            The circular detector is constructed to fit exactly into the
            Rowland Circle defined by ``rowland``.
        width : float
            Half-width of the tube in the flat direction (z-axis) in mm
        '''
        # Step 1: Get position and size from Rowland torus
        pos4d_circ = compose([rowland.R, 0, 0], np.eye(3), [rowland.r, rowland.r, width])
        # Step 2: Transform to global coordinate system
        pos4d_circ = np.dot(rowland.pos4d, pos4d_circ)
        # Step 3: Make detector
        return cls(pos4d=pos4d_circ, phi_offset=-np.pi)

    @property
    def _inwardsoutwards(self):
        'Transform the self.inwards bool into [-1, +1]'
        if self.inwards:
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
        # This could be moved to a general function
        if not np.all(dir[:, 3] == 0):
            raise ValueError('First input must be direction vectors.')
        # Could test pos, too...
        if transform:
            invpos4d = np.linalg.inv(self.pos4d)
            dir = np.dot(invpos4d, dir.T).T
            pos = np.dot(invpos4d, pos.T).T

        xyz = h2e(pos)

        # Solve quadratic equation in steps. a12 = (-xr +- sqrt(xr - r**2(x**2 - R**2)))
        xy = xyz[:, :2]
        r = dir[:, :2]
        underroot = (np.einsum('ij,ij->i', xy, r))**2 - np.sum(r**2, axis=1) * (np.sum(xy**2, axis=1) - 1.)
        intersect = (underroot >= 0)
        i = intersect  # just a shorthand because it's used so much below

        interpos_local = np.ones((pos.shape[0], 2))
        interpos_local[:] = np.nan
        interpos = np.ones_like(pos)
        interpos[:] = np.nan

        if intersect.sum() > 0:
            b = np.sum(xy[i] * r[i], axis=1)
            denom = np.sum(r[i]**2, axis=1)
            a1 = (- b + np.sqrt(underroot[i])) / denom
            a2 = (- b - np.sqrt(underroot[i])) / denom
            x1 = xy[i, :] + a1[:, np.newaxis] * r[i, :]
            apick = np.where(self._inwardsoutwards * np.sum(x1 * r[i, :], axis=1) >=0, a1, a2)
            xy_p = xy[i, :] + apick[:, np.newaxis] * r[i, :]
            phi = np.arctan2(xy_p[:, 1], xy_p[:, 0])
            # Shift phi by offset, then wrap to that it is in range [-pi, pi]
            interpos_local[i, 0] = (phi - self.phi_offset + np.pi) % (2 * np.pi) - np.pi
            # Those look like they hit in the xy plane.
            # Still possible to miss if z axis is too large.
            # Calculate z-coordiante at intersection
            interpos_local[intersect, 1] = xyz[i, 2] + apick * dir[i, 2]
            interpos[i, :2] = xy_p
            interpos[i, 2] = interpos_local[i, 1]
            interpos[i, 3] = 1
            # set those elements on intersect that miss in z to False
            trans, rot, zoom, shear = decompose44(self.pos4d)
            z_p = interpos[i, 2]
            intersect[i.nonzero()[0][np.abs(z_p) > 1]] = False
            # Now reset everything to nan that does not intersect
            interpos_local[~i, :] = np.nan
            # interpos_local in z direction is in local coordinates, i.e.
            # the x coordiante is 0..1, but we want that in units of the
            # global coordinate system.
            interpos_local[:, 1] = interpos_local[:, 1] * zoom[2]
            interpos[~i, :] = np.nan

            interpos = np.dot(self.pos4d, interpos.T).T

        return intersect, interpos, interpos_local

    def process_photons(self, photons, intersect, interpos, inter_local):
        photons['pos'][intersect, :] = interpos[intersect, :]
        self.add_output_cols(photons, self.loc_coos_name + self.detpix_name)
        # Add ID number to ID col, if requested
        if self.id_col is not None:
            photons[self.id_col][intersect] = self.id_num
        # Set position in different coordinate systems
        photons['pos'][intersect] = interpos[intersect]
        photons[self.loc_coos_name[0]][intersect] = inter_local[intersect, 0]
        photons[self.loc_coos_name[1]][intersect] = inter_local[intersect, 1]
        trans, rot, zoom, shear = decompose44(self.pos4d)
        if np.isclose(zoom[0], zoom[1]):
            photons[self.detpix_name[0]][intersect] = inter_local[intersect, 0] * zoom[0] / self.pixsize
        else:
            warnings.warn('Pixel coordinate for elliptical mirrors not implemented.', PixelSizeWarning)
        photons[self.detpix_name[1]][intersect] = inter_local[intersect, 1] / self.pixsize

        return photons

    def parametric_surface(self, phi, z=np.array([-1, 1])):
        '''Parametric description of the tube.

        This is just another way to obtain the shape of the tube, e.g.
        for visualization.

        Parameters
        ----------
        phi : np.array
            ``phi`` is the angle around the tube profile.
        z : np.array
            The coordiantes along the radius coordinate.

        Returns
        -------
        xyzw : np.array
            Ring coordinates in global homogeneous coordinate system.
        '''
        phi = np.asanyarray(phi)
        z = np.asanyarray(z)
        if (phi.ndim != 1) or (z.ndim != 1):
            raise ValueError('input parameters have 1-dim shape.')
        phi, z = np.meshgrid(phi, z)
        x = np.cos(phi)
        y = np.sin(phi)
        w = np.ones_like(z)
        coos = np.array([x, y, z, w]).T
        return np.einsum('...ij,...j', self.pos4d, coos)
