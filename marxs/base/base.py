# Licensed under GPL version 3 - see LICENSE.rst
from collections import OrderedDict
import inspect
import warnings
from copy import deepcopy

import numpy as np
from transforms3d import affines

from astropy.table import Column
from astropy.extern.six import with_metaclass

from ..visualization.utils import DisplayDict

class GeometryError(Exception):
    pass


class DocMeta(type):
    '''Metaclass to inherit docstrings when reqired.

    When a derived class overwrites a method that was already defined in its
    base class, the new method usually has the same purpose as the original method
    and often uses the same parameters, too, although the implementation differs
    slightly.
    In this case, it should have the same docstring, too.
    This metaclass will look for methods that are undocumented and add the docstring
    of the appropriate parent method to them.
    '''
    def __new__(mcs, name, bases, dict):
        # make a class here with the same method resolution order
        # but no attributes of its own (so that we can make it here without an
        # infinite loop, because making it will also go through this metaclass)
        if name == 'temporaryclass':
            return type.__new__(mcs, name, bases, dict)
        temp = type('temporaryclass', bases, {})
        mro = inspect.getmro(temp)

        for k in dict:
            # for items that have a docstring (i.e. methods) that is empty
            if hasattr(dict[k], '__doc__') and dict[k].__doc__ is None:
                # Some __doc__ strings cannot be rewitten in Python2, e.g. doc of a type
                try:
                    for b in mro:
                        # look if this b defines the method
                        # (it might not in case of multiple inheritance)
                        if hasattr(b, k) and (getattr(b, k).__doc__ is not None):
                            # copy docstring if super method has one
                            dict[k].__doc__ = getattr(b, k).__doc__
                            break
                except AttributeError:
                    pass

        return type.__new__(mcs, name, bases, dict)


class MarxsElement(with_metaclass(DocMeta, object)):
    '''Base class for all elements in a MARXS simulation.

    This includes elements that actually change photon properties such as grating and
    mirrors, but also abstract concepts that do not have a direct hardware
    representation such as a "Rowland Torus".
    '''

    display = {'shape': 'None'}
    'Dictionary for display specifications, e.g. color'

    def __init__(self, **kwargs):
        '''Define a new MARXS element.'''
        if 'name' in kwargs:
            self.name = kwargs.pop('name')
        else:
            self.name = self.__class__

        if len(kwargs) > 0:
            raise ValueError('Initialization arguments {0} not understood'.format(', '.join(kwargs.keys())))
        self.display = DisplayDict(self, deepcopy(self.display))

    def describe(self):
        return OrderedDict(element=self.name)

class SimulationSequenceElement(MarxsElement):
    '''Base class for all elements in a simulation sequence that processes photons.'''

    output_columns = []
    '''This is a list of strings that names the output properties.

    This gives the names of the output properties from this optical element.
    `process_photon` or `process_photons` are responsible for calculating the values of these
    properties. For example, for a mirror of nested shells one might set
    ``output_columns = ['mirror_shell']`` to pass the information on which shell the interaction
    took place to the user.

    The following properties are always included in the output and do not need to be listed here:

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
    '''

    id_col = None
    '''String that names an id column for output.

    Set this to a string to add an automatic numbering to the output. This is especially useful
    if there are several identical optical components that are used in parallel, e.g. there
    are four identical CCDs. Setting ``id_col = "CCD_ID"`` and passing an ``id_num=1, 2, 3, 4``
    keyword respectively to each CCD
    will add a column ``CCD_ID`` with a value of 1,2,3, or 4 for each photon hitting one of those
    CCDs.

    Currently, this will not work with all optical elements.
    '''

    def __init__(self, **kwargs):
        self.id_num = kwargs.pop('id_num', -9)
        # We want to use id_col as a class attribute, but overwrite it if given as a kwarg
        if 'id_col' in kwargs:
            self.id_col = kwargs.pop('id_col')
        super(SimulationSequenceElement, self).__init__(**kwargs)

    def add_output_cols(self, photons, colnames=[]):
        '''Add output columns of the correct format (currently: float) to the photon array.

        This function takes the column names that are added to ``photons`` from several sources:

        - `id_col` (if not ``None``)
        - `output_columns`
        - the ``colnames`` parameter.

        Parameters
        ----------
        photons : `astropy.table.Table`
            Table columns are added to.
        colnames : list of strings
            Column names to be added; in addition several object properties can be used to
            set the column names, see description above.
        '''
        temp = np.empty(len(photons))
        temp[:] = np.nan
        for n in self.output_columns + colnames:
            if n not in photons.colnames:
                photons.add_column(Column(name=n, data=temp))

        if self.id_col is not None:
            if self.id_col not in photons.colnames:
                photons.add_column(Column(name=self.id_col, data=-np.ones(len(photons))))

    def __call__(self, photons, *args, **kwargs):
        return self.process_photons(photons, *args, **kwargs)


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
