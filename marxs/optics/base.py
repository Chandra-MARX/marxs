from functools import wraps
from copy import copy

import numpy as np
from astropy.table import Table, Row

from ..math.pluecker import *
from ..base import SimulationSequenceElement, _parse_position_keywords

class OpticalElement(SimulationSequenceElement):
    '''Base class for all optical elements in marxs.

    This class cannot be used to instanciate an optical element directly, rather it serves as a
    base class from with other optical elements will be derived.

    At the very minumum, any derived class needs to implement either `process_photon` or
    `process_photons`. If the interaction with the photons (e.g. scattering of a mirror surface)
    can be implemented in a vectorized way using numpy array operations, the derived class should
    overwrite `process_photons` (`process_photon` is not used in ths case).
    If no vectorized implementation is available, it is sufficient to overwrite `process_photon`.
    Marxs will call `process_photons`, which (if not overwritten) contains a simple for-loop to
    loop over all photons in the array and call `process_photon` on each of them.
    '''

    geometry = {}
    '''A dictionary of geometric properties.

    Any entry in this dictionary that contains a 4-d array will be automatically transformed
    with the `pos4d` matrix when the object is initialized.

    For example, we might set ``geometry['center_of_mirror']=np.array([0, 0, 0, 1])``. When the user
    initializes an object of this type with the keyword ``position=[2,0,0]`` then the center
    of the new mirror object will be set to [2, 0, 0, 1].
    '''

    def __init__(self, **kwargs):
        self.pos4d = _parse_position_keywords(kwargs)

        # Before we change any numbers, we need to copy geometry from the class
        # attribute to an instance attribute
        self.geometry = copy(self.geometry)

        for elem, val in self.geometry.iteritems():
            if isinstance(val, np.ndarray) and (val.shape[-1] == 4):
                self.geometry[elem] = np.dot(self.pos4d, val)
        super(OpticalElement, self).__init__(**kwargs)

    def process_photon(self, dir, pos, energy, polarization):
        '''Simulate interaction of optical element with a single photon.

        Derived classes should overwrite this function or `process_photons`.

        Parameters
        ----------
        dir : `numpy.ndarray`
            4-d direction vector of ray in homogeneous coordinates
        pos : `numpy.ndarray`
            4-d position of last interaction pf the photons with any optical element in
            homogeneous coordinates. Together with ``dir`` this determines the equation
            of the ray.
        energy : float
            Photon energy in keV.
        polarization : float
            Polarization angle of the photons.

        Returns
        -------
        dir : `numpy.ndarray`
            4-d direction vector of ray in homogeneous coordinates
        pos : `numpy.ndarray`
            4-d position of last interaction pf the photons with any optical element in
            homogeneous coordinates. Together with ``dir`` this determines the equation
            of the ray.
        energy : float
            Photon energy in keV.
        polarization : float
            Polarization angle of the photons.
        probability : float
            Probability that the photon continues. Set to 0 if the photon is absorbed, to 1 if it
            passes the optical element and to number between 0 and 1 to express a probability that
            the photons passes.
        other : floats
            One number per entry in `output_columns`.
        '''
        raise NotImplementedError
        return dir, pos, energy, polarization, probability, any, other, output, columns

    def process_photons(self, photons):
        '''Simulate interaction of optical element with photons - vectorized.

        Derived classes should overwrite this function or `process_photon`.

        Parameters
        ----------
        photons: `astropy.table.Table` or `astropy.table.Row`
            Table with photon properties

        Returns
        -------
        photons: `astropy.table.Table` or `astropy.table.Row`
            Table with photon properties.
            If possible, the input table is modified in place, but in some
            cases this might not be possible and the returned Table may be
            a copy. Do not rely on either - use ``photons.copy()`` if you want
            to ensure you are working with an independent copy.
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

    '''

    geometry = {'center': np.array([0, 0, 0, 1.]),
                'v_y': np.array([0, 1., 0, 0]),
                'v_z': np.array([0, 0, 1., 0]),
                'v_x': np.array([1., 0, 0, 0]),
                'shape': 'box',
                }
    '''
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
    `pos4d` or in separate transformations (``position``, ``rotation``,
    ``zoom``).
    '''

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
            to ``np.nan`` is no intersecton point is found.
        interpos_local : `numpy.ndarray` of shape (N, 2)
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
        `astropy.table.Table`.
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
