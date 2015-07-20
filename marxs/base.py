from collections import OrderedDict

import numpy as np
from transforms3d import affines

class MarxsElement(object):
    '''Base class for all elements in a MARXS simulation.
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

class SimulationSequenceElement(MarxsElement):
    '''Base class for all elements in a simulation sequence.'''
    pass

def _parse_position_keywords(kwargs):
    '''Parse keywords to define position.

    If ``pos4d`` is given, use that, otherwise look for ``position``, ``zoom``
    and ``orientation``.

    Parameters
    ----------
    pos4d : 4x4 array
        Transformation to bring an element from the default position (see description of
        individual elements) to a certain point in the space of homogeneous coordinates.
    position : 3-d vector in real space
        Measured from the origin of the spacecraft coordinate system.
    orientation : Rotation matrix or ``None``
        Relative orientation of the base vectors of this optical
        element relative to the orientation of the coordinate system
        of the spacecraft. The default is no rotation (i.e. the axes of both
        coordinate systems are parallel).
    zoom : float or 3-d vector
        Scale the size of an optical element in each dimension by ``zoom``. If ``zoom`` is a scalar,
        the same scale is applied in all dimesions. This only affects the oter dimensions of
        the optical element, not internal scales like the pixel size or grating constant (if
        defined for the optical element in question).
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
