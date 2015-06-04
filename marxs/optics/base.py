# import abc  - do I need this?
from collections import OrderedDict
from functools import wraps
from copy import copy

import numpy as np
from transforms3d import affines
from astropy.table import Table, Row

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
        element relative to the orientation of the coordinate system
        of the spacecraft. The default is no rotation (i.e. the axes of the
        coordinate systems are parallel).
    '''
    # __metaclass__ = abc.ABCMeta

    geometry = {}
    name = ''
    output_columns = []

    def __init__(self, **kwargs):
        self.pos4d = kwargs.pop('pos4d', None)
        if self.pos4d is None:
            self.position = kwargs.pop('position', np.zeros(3))
            self.orientation = kwargs.pop('orientation', np.eye(3))
            self.zoom = kwargs.pop('zoom', 1.)
            if np.isscalar(self.zoom):
                self.zoom = np.ones(3) * self.zoom
            self.pos4d = affines.compose(self.position, self.orientation, self.zoom)

        # Before we change any numbers, we need to copy geometry from the class
        # attribute to an instance attribute
        self.geometry = copy(self.geometry)

        for elem, val in self.geometry.iteritems():
            self.geometry[elem] = np.dot(self.pos4d, val)
        super(OpticalElement, self).__init__(**kwargs)

    def add_output_columns(self, photons):
        for col in self.output_columns:
            photons.add_empty_column

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
        self.add_output_columns(photons)
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
    geometry = {'center': np.array([0, 0, 0, 1]),
                'e_y': np.array([0, 1, 0, 0]),
                'e_z': np.array([0, 0, 1, 0]),
                }

    def __init__(self, **kargs):
        super(FlatOpticalElement, self).__init__(**kargs)
        normal = e2h(np.cross(h2e(self.geometry['e_y']), h2e(self.geometry['e_z'])), 0)
        self.geometry['plane'] = point_dir2plane(self.geometry['center'],
                                                 normal)

    def intersect(self, dir, pos):
        plucker = dir_point2line(h2e(dir), h2e(pos))
        return intersect_line_plane(plucker, self.geometry['plane'])


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
        The function to be decorated. In the signature, ``photons`` is an
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
