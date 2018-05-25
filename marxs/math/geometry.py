# Licensed under GPL version 3 - see LICENSE.rst
from functools import wraps
from copy import copy

import numpy as np
from astropy.table import Table, Row
from transforms3d.affines import decompose44, compose

from .utils import e2h, h2e
from .pluecker import point_dir2plane
from ..base import SimulationSequenceElement, _parse_position_keywords


class BaseGeometry(object):
    pass


class Geometry(BaseGeometry):
    _geometry = {}
    _display = {}

    def __init__(self, kwargs={}):
        self.pos4d = _parse_position_keywords(kwargs)
        #copy class attribute to instance attribute
        self._geometry = copy(self._geometry)


    def __getitem__(self, key):
        '''This function wraps access to the pos4d matrix.

        This is mostly a convenience method that gives access to vectors from the
        ``pos4d`` matrix in familiar terms with string labels:

        - ``center``: The ``center`` is the origin of the local coordiante system
          of the optical elemement. Typically, if will be the center of the
          active plane, e.g. the center of the mirror surface.
        - :math:`\hat v_{y,z}`: The box stretches in the y and z direction for
          :math:`\pm \hat v_y` and :math:`\pm \hat v_z`. In optics, the relevant
          interaction often happens on a surface, e.g. the surface of a mirror or
          of a reflection grating. In the defaul configuration, this surface is in
          the yz-plane.
        - :math:`\hat v_x`: The thickness of the box is not as important in many
          cases,
          but is useful to e.g. render the geometry or to test if elements collide.
          The box reaches from the yz-plane down to :math:`- \hat v_x`. Note, that
          this definition means the size parallel to the y and z axis is
          :math:`2 |\hat v_{y,z}|`, but only :math:`1 |\hat v_{x}|`.
        - :math:`\hat e_{x,y,z}`: When an object is initialized, it automatically
          adds unit vectors in the
          direction of :math:`\hat v_{x,y,z}` called :math:`\hat e_{x,y,z}`.
        - ``plane``: It also adds the homogeneous coordinates of the active plane
          as ``plane``.
        - Other labels get looked up in ``self._geometry`` and if the resulting
          value is a 4-d vector, it gets transformed with ``pos4d``.

        Not all these labels make sense for every optical element (e.g. a curved
        mirror does not really have a "plane").
        Access through this method is slower than direct indexing of ``self.pos4d``.
        '''

        if key == 'center':
            return self.pos4d[:, 3]
        elif key in ['v_x', 'e_x']:
            val = self.pos4d[:, 0]
        elif key in ['v_y', 'e_y']:
            val = self.pos4d[:, 1]
        elif key in ['v_z', 'e_z']:
            val = self.pos4d[:, 2]
        elif key == 'plane':
            return point_dir2plane(self['center'], self['e_x'])
        else:
            val = self._geometry[key]
            if isinstance(val, np.ndarray) and (val.shape[-1] == 4):
                val = np.dot(self.pos4d, val)
        if key[:2] == 'e_':
            return val / np.linalg.norm(val)
        else:
            return val

    def intersect(self, dir, pos):
        '''Calculate the intersection point between a ray and the element

        Parameters
        ----------
        dir : `numpy.ndarray` of shape (N, 4)
            homogeneous coordinates of the direction of the ray
        pos : `numpy.ndarray` of shape (N, 4)
            homogeneous coordinates of a point on the ray

        Returns
        -------
        intersect :  boolean array of length N
            ``True`` if an intersection point is found.
        interpos : `numpy.ndarray` of shape (N, 4)
            homogeneous coordinates of the intersection point. Values are set
            to ``np.nan`` if no intersection point is found.
        interpos_local : `numpy.ndarray` of shape (N, 2)
            y and z coordinates in the coordiante system of the active plane.
        '''
        raise NotImplementedError

    def display(self, key):
        if key in self._display:
            return self._display[key]
        else:
            return super(Geometry, self).display(key)


class FinitePlane(Geometry):
    '''Base class for geometrically flat optical elements.

    '''

    _display = {'shape': 'box'}

    loc_coos_name = ['y', 'z']
    '''name for output columns that contain the interaction point in local coordinates.'''

    def intersect(self, dir, pos):
        '''Calculate the intersection point between a ray and the element

        Parameters
        ----------
        dir : `numpy.ndarray` of shape (N, 4)
            homogeneous coordinates of the direction of the ray
        pos : `numpy.ndarray` of shape (N, 4)
            homogeneous coordinates of a point on the ray

        Returns
        -------
        intersect :  boolean array of length N
            ``True`` if an intersection point is found.
        interpos : `numpy.ndarray` of shape (N, 4)
            homogeneous coordinates of the intersection point. Values are set
            to ``np.nan`` if no intersection point is found.
        interpos_local : `numpy.ndarray` of shape (N, 2)
            y and z coordinates in the coordiante system of the active plane
            (not normalized to the dimensions of the element in question, but
            in absolute units).
        '''
        k_nominator = np.dot(self['center'] - pos, self['e_x'])
        k_denom = np.dot(dir, self['e_x'])

        is_parallel = k_denom == 0
        # To avoid warning for parallel rays, which is handeled expicitly below
        with np.errstate(divide='ignore'):
            k = k_nominator / k_denom

        forward = k >= 0  # with k < 0, rays would have to move backwards.

        if dir.ndim == 2:
            interpos = pos + k[:, None] * dir  # broadcasting array
        else:
            interpos = pos + k * dir           # dir is a scalar
        vec_center_inter = interpos - self['center']
        interpos_local = np.vstack([np.dot(vec_center_inter, self['e_y']),
                                    np.dot(vec_center_inter, self['e_z'])]).T

        intersect = (~is_parallel & forward &
                     (np.abs(interpos_local[:, 0]) <= np.linalg.norm(self['v_y'])) &
                     (np.abs(interpos_local[:, 1]) <= np.linalg.norm(self['v_z'])))
        for i in [interpos, interpos_local]:
            if dir.ndim == 2:
                i[~intersect, :3] = np.nan
            # input of single photon.
            elif (dir.ndim == 1) and not intersect:
                i[:3] = np.nan

        return intersect, interpos, interpos_local


class Cylinder(Geometry):
    '''A detector shaped like a ring or tube.

    This detector is shaped like a tube. The form is a circle in the xy plane
    and and flat along the z-direction.  While most CCDs are flat in practice,
    the `CircularDetector` simulates a setup that can follow the Rowland circle
    geometry exactly which is useful, e.g. to study the resolution of a
    spectrograph without worrying about the details of the detector geometry.

    Parameters
    ----------
    position, orientation, zoom, pos4d : see description of `pos4d`
        The radius of the tube is given by the ``zoom`` keyword, see `pos4d`.
        Use ``zoom[0] == zoom[1]`` to make a circular tube. ``zoom[0] != zoom[1]`` gives
        an elliptical profile. ``zoom[2]`` sets the extension in the z direction.
    pixsize : float
        size of pixels in mm
    '''
    loc_coos_name = ['phi', 'y']


    _display = {'shape': 'surface',
               'coo1': np.linspace(-np.pi, np.pi, 50),
               'coo2': [-1, 1]}

    @property
    def phi_lim(self):
        return self._phi_lim

    @phi_lim.setter
    def phi_lim(self, value):
        if (len(value) != 2) or (value[0] > value[1]):
            raise ValueError('phi_lim has the format [lower limit, upper limit]')
        for v in value:
            if (v < -np.pi) or (v > np.pi):
                raise ValueError('phi_lim must be in range -pi to +pi.')
        self._phi_lim = value
        self._display['coo1'] = np.linspace(value[0], value[1], 50)

    def __init__(self, kwargs={}):
        self.phi_lim = kwargs.pop('phi_lim', [-np.pi, np.pi])
        super(Cylinder, self).__init__(kwargs)

    def __getitem__(self, value):
        if value == 'R':
            trans, rot, zoom, shear = decompose44(self.pos4d)
            return zoom[0]
        else:
            return super(Cylinder, self).__getitem__(value)

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
        return cls({'pos4d': pos4d_circ, 'phi_offset': np.pi})


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
        dir_e = h2e(dir)

        # Solve quadratic equation in steps. a12 = (-xr +- sqrt(xr - r**2(x**2 - R**2)))
        xy = xyz[:, :2]
        r = dir[:, :2]
        underroot = (np.einsum('ij,ij->i', xy, r))**2 - np.sum(r**2, axis=1) * (np.sum(xy**2, axis=1) - 1.)
        # List of intersect in xy plane.
        intersect = (underroot >= 0)
        i = intersect  # just a shorthand because it's used so much below

        interpos_local = np.ones((pos.shape[0], 2))
        interpos_local[:] = np.nan
        interpos = np.ones_like(pos)
        interpos[:] = np.nan

        if intersect.sum() > 0:
            i_ind = intersect.nonzero()[0]
            b = np.sum(xy[i] * r[i], axis=1)
            denom = np.sum(r[i]**2, axis=1)
            a1 = (- b + np.sqrt(underroot[i])) / denom
            a2 = (- b - np.sqrt(underroot[i])) / denom
            xy_1 = xy[i, :] + a1[:, np.newaxis] * r[i, :]
            phi_1 = np.arctan2(xy_1[:, 1], xy_1[:, 0])
            xy_2 = xy[i, :] + a2[:, np.newaxis] * r[i, :]
            phi_2 = np.arctan2(xy_2[:, 1], xy_2[:, 0])
            # 1, 2 look like hits in x,y but might still miss in z
            z_1 = xyz[i, 2] + a1 * dir[i, 2]
            z_2 = xyz[i, 2] + a2 * dir[i, 2]
            hit_1 = ((a1 >= 0) & (np.abs(z_1) <= 1.) &
                     (phi_1 >= self.phi_lim[0]) & (phi_1 <= self.phi_lim[1]))
            # Use hit_2 only if a2 is closer than hit_1
            hit_2 = ((a2 >= 0) & (a2 <= a1) & (np.abs(z_2) <= 1.) &
                     (phi_2 >= self.phi_lim[0]) & (phi_2 <= self.phi_lim[1]))
            intersect[i_ind] = hit_1 | hit_2
            # Set values into array from either point 1 or 2
            interpos_local[i_ind[hit_1], 0] = phi_1[hit_1]
            interpos_local[i_ind[hit_1], 1] = z_1[hit_1]

            interpos_local[i_ind[hit_2], 0] = phi_2[hit_2]
            interpos_local[i_ind[hit_2], 1] = z_2[hit_2]
            # Calculate pos for point 1 or 2 in local xyz coord system
            interpos[i_ind[hit_1], :] = e2h(xyz[i_ind, :] + a1[:, None] * dir_e[i_ind, :], 1)[hit_1, :]
            interpos[i_ind[hit_2], :] = e2h(xyz[i_ind, :] + a2[:, None] * dir_e[i_ind, :], 1)[hit_2, :]


            trans, rot, zoom, shear = decompose44(self.pos4d)
            # interpos_local in z direction is in local coordinates, i.e.
            # the x coordiante is 0..1, but we want that in units of the
            # global coordinate system.
            interpos_local[:, 1] = interpos_local[:, 1] * zoom[2]
            interpos = np.dot(self.pos4d, interpos.T).T

        return intersect, interpos, interpos_local


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
