# import abc  - do I need this?
from collections import OrderedDict
from functools import wraps
from copy import copy

import numpy as np
from transforms3d import affines
from astropy.table import Table, Row, Column

from ..math.pluecker import *


class SimulationSequenceElement(object):
    '''Base element for everything that's allowed in a sequence. Could go into
    a more general file once it gets more than 5 lines as it's also used for
    e.g. source and maybe also filters, files handles?
    '''
    def __init__(self, **kwargs):
        if 'name' in kwargs:
            self.name = kwargs.pop('name')
        else:
            self.name = self.__class__

        if len(kwargs) > 0:
            raise ValueError('Initialization arguments {0} not understood'.format(', '.join(kwargs.keys())))

    def describe(self):
        return OrderedDict(element=self.name)


def _parse_position_keywords(kwargs):
    '''Parse keywords to define position.

    If pos4d is given, use that, otherwise look for ``position``, ``zoom``
    and ``orientation``.
    '''
    pos4d = kwargs.pop('pos4d', None)
    if pos4d is None:
        position = kwargs.pop('position', np.zeros(3))
        orientation = kwargs.pop('orientation', np.eye(3))
        zoom = kwargs.pop('zoom', 1.)
        if np.isscalar(zoom):
            zoom = np.ones(3) * zoom
        if not len(zoom) == 3:
            raise ValueError('zoom must have three elements for x,y,z or be a scalar (global zoom).')
        pos4d = affines.compose(position, orientation, zoom)
    else:
        if ('position' in kwargs) or ('orientation' in kwargs) or ('zoom' in kwargs):
            raise ValueError('If pos4d is specificed, the following keywords cannot be given at the same time: position, orientation, zoom.')
    return pos4d

class OpticalElement(SimulationSequenceElement):
    '''
    Parameters
    ----------
    pos4d : 4x4 array
        takes precedence over ``position`` and ``orientation``
    position : 3-d vector in real space
        Measured from the origin of the spacecraft coordinate system
    orientation : Rotation matrix on ``None``
        Relative orientation of the base vectors of this optical
        element relative to the orientation of the cooxsrdinate system
        of the spacecraft. The default is no rotation (i.e. the axes of the
        coordinate systems are parallel).
    '''
    # __metaclass__ = abc.ABCMeta

    geometry = {}
    name = ''
    output_columns = []

    def __init__(self, **kwargs):
        self.pos4d = _parse_position_keywords(kwargs)

        # Before we change any numbers, we need to copy geometry from the class
        # attribute to an instance attribute
        self.geometry = copy(self.geometry)

        for elem, val in self.geometry.iteritems():
            if isinstance(val, np.ndarray) and (val.shape[-1] == 4):
                self.geometry[elem] = np.dot(self.pos4d, val)
        super(OpticalElement, self).__init__(**kwargs)

    def add_output_cols(self, photons):
        temp = np.empty(len(photons))
        temp[:] = np.nan
        for n in self.output_columns:
            if n not in photons.colnames:
                photons.add_column(Column(name=n, data=temp))

    def process_photon(self, dir, pos, energy, polarization):
        raise NotImplementedError

    def process_photons(self, photons):
        '''
        Parameters
        ----------
        photons: `~astropy.table.Table` or `~astropy.table.Row`
            Table with photon properties

        Returns
        -------
        photons: `~astropy.table.Table` or `~astropy.table.Row`
            Table with photon properties.
            If possible, the input table is modified in place, but in some
            cases this might not be possible and the returned Table may be
            a copy. Do not rely on either - use ``photons.copy()`` if you want
            to ensure you are working with an independent copy.

        This is the simple and naive and probably slow implementation. For
        performance, I might want to pull out the relevant numpy arrays ahead
        of time and then iterate over those, because that can be optimized by
        e.g. numba (I don't think that numba can enhance the entire
        astropy.table package) - but that is for a later time.
        '''
        if isinstance(photons, Row):
            photons = Table(photons)
        outcols = ['dir', 'pos', 'energy', 'polarization', 'probability'] + self.output_columns
        for i, photon in enumerate(photons):
            outs = self.process_photon(photon['dir'], photon['pos'],
                                       photon['energy'],
                                       photon['polarization'])
            for a, b in zip(outcols, outs):
                if a == 'probability':
                    photons['probability'][i] *= b
                else:
                    photons[a][i] = b
        return photons



class FlatOpticalElement(OpticalElement):
    '''Base class for geometrically flat optical elements.

    Each flat optical element contains a dictionary called ``geometry`` that
    describes its geometrical properties. This dictionary has the following
    entries:

    - ``shape``: The default geometry has the ``shape`` of a box.
    - ``center``: The ``center`` is the origin of the local coordiante system
      of the optical elemement. Typically, if will be the center of the
      active plane, e.g. the center of the mirror surface. Initially, the
      ``center`` of the element is at the origin of the global coordiante
      system.
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

    When an object is initialized, all the above entries are transformed with
    its 4-dim transformation matrix, that can either be passed directly as
    ``pos4d`` or in separate transformations (``position``, ``rotation``,
    ``zoom``).
    '''
    geometry = {'center': np.array([0, 0, 0, 1.]),
                'v_y': np.array([0, 1., 0, 0]),
                'v_z': np.array([0, 0, 1., 0]),
                'v_x': np.array([1., 0, 0, 0]),
                'shape': 'box',
                }

    def __init__(self, **kwargs):
        super(FlatOpticalElement, self).__init__(**kwargs)
        for c in 'xyz':
            self.geometry['e_' + c] = self.geometry['v_' + c] / np.linalg.norm(self.geometry['v_' + c])
        normal = e2h(np.cross(h2e(self.geometry['e_y']), h2e(self.geometry['e_z'])), 0)
        self.geometry['plane'] = point_dir2plane(self.geometry['center'],
                                                 normal)

    def intersect(self, dir, pos):
        '''Calculate the intersection point between a ray and the element

        Parameters
        ----------
        dir : np.array of shape (N, 4)
            homogeneous coordinates of the direction of the ray
        pos : np.array of shape (N, 4)
            homogeneous coordinates of a point on the ray

        Returns
        -------
        intersect :  boolean array of length N
            ``True`` if an intersection point is found.
        interpos : np.array of shape (N, 4)
            homogeneous coordinates of the intersection point. Values are set
            to ``np.nan`` is no intersecton point is found.
        interpos_local : np.array of shape (N, 2)
            y and z coordinates in the coordiante system of the active plane.
        '''
        plucker = dir_point2line(h2e(dir), h2e(pos))
        interpos =  intersect_line_plane(plucker, self.geometry['plane'])
        vec_center_inter = - h2e(self.geometry['center']) + h2e(interpos)
        ey = np.dot(vec_center_inter, h2e(self.geometry['e_y']))
        ez = np.dot(vec_center_inter, h2e(self.geometry['e_z']))
        intersect = ((np.abs(ey) <= np.linalg.norm(self.geometry['v_y'])) &
                     (np.abs(ez) <= np.linalg.norm(self.geometry['v_z'])))
        if dir.ndim == 2:
            interpos[~intersect, :3] = np.nan
        # input of single photon.
        elif (dir.ndim == 1) and not intersect:
            interpos[:3] = np.nan
        return intersect, interpos, np.vstack([ey, ez]).T


def photonlocalcoords(f, colnames=['pos', 'dir']):
    '''Decorator for calculation that require a local coordinate system

    This is specifically meant to wrap the :meth:`process_photons` methods of
    any :class:`OpticalElement`; the current implementation expects the call
    signature of :meth:`process_photons`.

    This decorator transforms coordinates from the global system to the local
    system before a function call and back to the global system again after
    the function call.

    Parameters
    ----------
    f : callable with signature ``f(self, photons, *args, **kwargs)``
        The function to be decorated. In the signature, ``photons`` is a
        `~astropy.table.Table`.
    colnames : list of strings
        List of all column names in the photon table to be transformed into a
        different coordinate system.
    '''
    @wraps(f)
    def wrapper(self, photons, *args, **kwargs):
        # transform to coordsys if single instrument
        invpos4d = np.linalg.inv(self.pos4d)
        for n in colnames:
            photons[n] = np.einsum('...ij,...j', invpos4d, photons[n])
        photons = f(self, photons, *args, **kwargs)
        # transform back into coordsys of satellite
        for n in colnames:
            photons[n] = np.einsum('...ij,...j', self.pos4d, photons[n])
        return photons

    return wrapper
