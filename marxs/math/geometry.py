# Licensed under GPL version 3 - see LICENSE.rst
from copy import copy, deepcopy

import numpy as np
from transforms3d.affines import decompose44, compose
from transforms3d.axangles import axangle2mat

from .utils import e2h, h2e, anglediff
from .pluecker import point_dir2plane
from ..base import _parse_position_keywords
from ..visualization.utils import plane_with_hole, combine_disjoint_triangulations


def xyz_square(geometry, r_factor=1):
    '''Generate Eukledian positions for the corners of a square.

    The square is centered on the center of the object and the edges are
    given by ``v_y`` and ``v_z``.

    Parameters
    ----------
    r_factor : float
        Scaling factor for the square.

    Returns
    -------
    box : np.array of shape (4, 3)
        Eukledian coordinates of the corners of the square in 3d space.
    '''
    g = geometry
    box = h2e(g['center']) + r_factor * np.vstack([h2e( g['v_y']) + h2e(g['v_z']),
                                                   h2e(-g['v_y']) + h2e(g['v_z']),
                                                   h2e(-g['v_y']) - h2e(g['v_z']),
                                                   h2e( g['v_y']) - h2e(g['v_z'])
                                               ])
    return box

def xyz_circle(geometry, r_factor=1, philim=[0, 2 * np.pi], n_vertices=90):
    '''Generate Eukledian positions along an ellipse.

    The circle is centered on the center of the object and the semi-major
    and minor axes are given by ``v_y`` and ``v_z``. Note that this function
    is usually used to generate circle position, although ellipses are possible,
    thus the name.

    The circle (or ellipse) is approximated by a polygon with ``n_vertices``
    vertices, where the value of ``n_vertices`` is taken from the ``self.display``
    dictionary.

    Parameters
    ----------
    r_factor : float
        Scaling factor for the square.
    phi_lim : list
        Lower and upper limit for the angle phi to restrict the circle to a wedge.

    Returns
    -------
    circle : np.array of shape (n, 3)
        Eukledian coordinates of the corners of the square in 3d space.
    '''
    n = n_vertices
    phi = np.linspace(0.5 * np.pi, 2.5 * np.pi, n, endpoint=False)
    v_y = r_factor * geometry['v_y']
    v_z = r_factor * geometry['v_z']

    x = np.cos(phi)
    y = np.sin(phi)
    # phi could be less then full circle
    # wrap phi at lower bound (which could be negative).
    # For the default [0, 2 pi] this is a no-op
    phi = (phi - philim[0]) % (2 * np.pi)
    ind = phi < anglediff(philim)
    x[~ind] = 0
    y[~ind] = 0

    return h2e(geometry['center'] + x.reshape((-1, 1)) * v_y + y.reshape((-1, 1)) * v_z)


class Geometry(object):
    _geometry = {}

    shape = 'None'
    n_points = 50

    def __init__(self, kwargs={}):
        self.pos4d = _parse_position_keywords(kwargs)
        #copy class attribute to instance attribute
        self._geometry = copy(self._geometry)
        self.geometry = self


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


class FinitePlane(Geometry):
    '''Base class for geometrically flat optical elements.

    '''

    shape = 'box'

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


class PlaneWithHole(FinitePlane):

    shape='triangulation'
    outer_factor = 3
    inner_factor = 0

    def __init__(self, kwargs):
        self._geometry['r_inner'] = kwargs.pop('r_inner', 0.)
        super(PlaneWithHole, self).__init__(kwargs)

    def triangulate(self, display={}):
        '''Return a triangulation of the aperture hole embedded in a square.

        The size of the outer square is determined by the ``'outer_factor'`` element
        in ``self.display``.

        Returns
        -------
        xyz : np.array
            Numpy array of vertex positions in Eukeldian space
        triangles : np.array
            Array of index numbers that define triangles
        '''
        outer_disp = self.outer_display(display)
        outer_shape = self.outer_shape(display)

        xyz, triangles = plane_with_hole(outer_disp, outer_shape)

        if self['r_inner'] > 0.:
            inner_shape = self.inner_shape(display)
            # Inner edge of the display. If we have several stacked apertures,
            # we don't want to fill is all up to r=0.
            inner_disp = self.inner_display(display)
            new_xyz, new_tri = plane_with_hole(inner_shape, inner_disp)
            xyz, triangles = combine_disjoint_triangulations([xyz, new_xyz],
                                                             [triangles, new_tri])

        return xyz, triangles

    def outer_shape(self, diplay):
        raise NotImplementedError

    def outer_display(self, diplay):
        raise NotImplementedError

    def inner_shape(self, diplay):
        raise NotImplementedError

    def inner_display(self, diplay):
        raise NotImplementedError


class RectangleHole(PlaneWithHole):
    def outer_shape(self, display):
        return xyz_square(self)

    def outer_display(self, display):
        return xyz_square(self, r_factor=display['outer_factor'])


class CircularHole(PlaneWithHole):

    def __init__(self, kwargs):
        phi = kwargs.pop('phi', [0, 2. * np.pi])
        if np.max(np.abs(phi)) > 10:
            raise ValueError('Input angles >> 2 pi. Did you use degrees (radian expected)?')
        if phi[0] > phi[1]:
            raise ValueError('phi[1] must be greater than phi[0].')
        if (phi[1] - phi[0]) > (2 * np.pi + 1e-6):
            raise ValueError('phi[1] - phi[0] must be less than 2 pi.')
        self.phi = phi

        super(CircularHole, self).__init__(kwargs)

    def outer_display(self, display):
        '''Return values in Eukledian space.'''
        return xyz_circle(self, r_factor=display['outer_factor'])

    def outer_shape(self, display):
        return xyz_circle(self, r_factor=1, philim=self.phi)

    def inner_shape(self, display):
        return xyz_circle(self,
                          r_factor=self['r_inner']/np.linalg.norm(self['v_y']),
                          philim=self.phi)
    def inner_display(self, display):
        # Inner edge of the display. If we have several stacked apertures,
        # we don't want to fill is all up to r=0.
        return xyz_circle(self,
                          r_factor=display['inner_factor'] * self['r_inner'] / np.linalg.norm(self['v_y']),
                          philim=self.phi)

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
    phi_lim : list
        If a cylinder does not cover the full circle, set ``phi_lim`` to the limits, e.g.
        ``[-np.pi / 2, np.pi / 2]`` makes a "half-pipe".
    '''
    loc_coos_name = ['phi', 'y']

    shape = 'surface'
    coos_limits = [np.array([-np.pi, np.pi]), np.array([-1, 1])]

    # This verification code does not fit here right now, but when I convert to traits
    # later it might come in useful again and thus I keep it here as a comment.
    # @property
    # def phi_lim(self):
    #     return self._phi_lim

    # @phi_lim.setter
    # def phi_lim(self, value):
    #     if (len(value) != 2) or (value[0] > value[1]):
    #         raise ValueError('phi_lim has the format [lower limit, upper limit]')
    #     for v in value:
    #         if (v < -np.pi) or (v > np.pi):
    #             raise ValueError('phi_lim must be in range -pi to +pi.')
    #     self.coos_limits[0] = value

    def __init__(self, kwargs={}):
        self.coos_limits = deepcopy(self.coos_limits)
        self.coos_limits[0] = np.asanyarray(kwargs.pop('phi_lim', [-np.pi, np.pi]))
        super(Cylinder, self).__init__(kwargs)

    def __getitem__(self, value):
        if value == 'R':
            trans, rot, zoom, shear = decompose44(self.pos4d)
            return zoom[0]
        else:
            return super(Cylinder, self).__getitem__(value)

    @classmethod
    def from_rowland(cls, rowland, width, rotation=0., kwargs={}):
        '''Generate a `Cylinder` from a `RowlandTorus`.

        According to the definition of the `marxs.design.rowland.RowlandTorus`
        the origin phi=0 is at the "top". When this class method is used to
        make a detector that catches all dispersed grating signal on the
        Rowland torus, a ``rotation=np.pi`` places the center of the Cylinder
        close to the center of the torus (the location of the focal point in
        the standard Rowland geometry).

        Parameters
        ----------
        rowland : `~marxs.design.RowlandTorus`
            The circular detector is constructed to fit exactly into the
            Rowland Circle defined by ``rowland``.
        width : float
            Half-width of the tube in the flat direction (z-axis) in mm
        rotation : float
            Rotation angle of the Cylinder around its z-axis compared to the
            phi=0 position of the Rowland torus.

        '''
        # Step 1: Rotate around z axis
        rot = axangle2mat(np.array([0, 0, 1.]), rotation)
        # Step 2: Get position and size from Rowland torus
        pos4d_circ = compose([rowland.R, 0, 0], rot, [rowland.r, rowland.r, width])
        # Step 3: Transform to global coordinate system
        pos4d_circ = np.dot(rowland.pos4d, pos4d_circ)
        # Step 4: Make detector
        det_kwargs = {'pos4d': pos4d_circ}
        det_kwargs.update(kwargs)
        return cls(det_kwargs)


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
                     (phi_1 >= self.coos_limits[0][0]) & (phi_1 <= self.coos_limits[0][1]))
            # Use hit_2 only if a2 is closer than hit_1
            hit_2 = ((a2 >= 0) & (a2 <= a1) & (np.abs(z_2) <= 1.) &
                     (phi_2 >= self.coos_limits[0][0]) & (phi_2 <= self.coos_limits[0][1]))
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


    def parametric_surface(self, phi=None, z=None, display={}):
        '''Parametric description of the tube.

        This is just another way to obtain the shape of the tube, e.g.
        for visualization.

        Parameters
        ----------
        phi : np.array
            ``phi`` is the angle around the tube profile. Set to ``None`` to use the
            extend of the element itself.
        z : np.array
            The coordiantes along the radius coordinate. Set to ``None`` to use the
            extend of the element itself.

        Returns
        -------
        xyzw : np.array
            Ring coordinates in global homogeneous coordinate system.
        '''
        phi = np.linspace(self.coos_limits[0][0], self.coos_limits[0][1], self.n_points) \
              if phi is None else np.asanyarray(phi)
        z = self.coos_limits[1] if z is None else np.asanyarray(z)
        if (phi.ndim != 1) or (z.ndim != 1):
            raise ValueError('input parameters have 1-dim shape.')
        phi, z = np.meshgrid(phi, z)
        x = np.cos(phi)
        y = np.sin(phi)
        w = np.ones_like(z)
        coos = np.array([x, y, z, w]).T
        return np.einsum('...ij,...j', self.pos4d, coos)
