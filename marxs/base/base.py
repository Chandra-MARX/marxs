# Licensed under GPL version 3 - see LICENSE.rst
from collections import OrderedDict
import inspect
from copy import deepcopy
import re
from datetime import datetime

import numpy as np
from transforms3d import affines
from astropy.table import Column

from ..visualization.utils import DisplayDict
from marxs import __version__

__all__ = ['GeometryError',
           'DocMeta',
           'MarxsElement',
           'TagVersion',
           'check_meta_consistent',
           'check_energy_consistent',
           'SimulationSequenceElement',
           ]


class GeometryError(Exception):
    pass


class DocMeta(type):
    '''Metaclass to inherit docstrings when required.

    When a derived class overwrites a method that was already defined in its
    base class, the new method usually has the same purpose as the original
    method and often uses the same parameters, too, although the implementation
    differs slightly.  In this case, it should have the same docstring, too.
    This metaclass will look for methods that are undocumented and add the
    docstring of the appropriate parent method to them.

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


class MarxsElement(metaclass=DocMeta):
    '''Base class for all elements in a MARXS simulation.

    This includes elements that actually change photon properties such
    as grating and mirrors, but also abstract concepts that do not
    have a direct hardware representation such as a "Rowland Torus".
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


reexp = re.compile(r"(?P<version>[\d.dev]+[\d]+)[+]?(g(?P<gittag>\w+))?[.]?(d(?P<dirtydate>[\d]+))?")
'''Regex to parse scm version string'''

ver = reexp.match(__version__)
'''Parsed version of the marxs code'''


class TagVersion(MarxsElement):
    '''Tag a photons list with diagnostic information such as a the program version.

    All keyword arguments passed when this element is initialized or when it is called
    will be added to the meta information of the photon list, some additional
    information on program version and runtime is added automatically.
    As such, the format of the keyword values is very flexible. However, if they are
    going to be written to fits files it is useful to follow fits conventions and
    the default paraemter values also invoke fits conventions.


    Parameters
    ----------
    origin : tuple of strings
        according to fits convention, the Institution where file was created
    creator : string or tuple of strings
        according to fits convention, the Person or program creating file'
    '''
    def __init__(self,
                 ORIGIN=('unkwown', 'Institution where file was created'),
                 CREATOR=('MARXS', 'Person or program creating file'),
                 **kwargs):
        super().__init__(name=kwargs.pop('name', self.__class__))

        kwargs['MARXSVER'] = (ver.group('version'), 'MARXS code version')
        if not ver.group('gittag') is None:
            kwargs['MARXSGIT'] = (ver.group('gittag'),
                                        'Git hash of MARXS code')
        if not ver.group('dirtydate') is None:
            kwargs['MARXSTIM'] = (ver.group('dirtydate'),
                                        'Date of dirty version ARCUS code')
        self.tags = kwargs

    def __call__(self, photons, *args, **kwargs):
        photons.meta['DATE'] = (datetime.now().isoformat()[:10], 'Date/time of computation')
        for k, v in self.tags.items():
            photons.meta[k] = v
        for k, v in kwargs.items():
            photons.meta[k] = v
        return photons


def check_meta_consistent(meta1, meta2, keywords=['ORIGIN', 'CREATOR',
                                                  'MARXSVER', 'MARXSGIT', 'MARXSTIM',
                                                  ],
                          allow_missing=True):
    '''Check that the meta data between two simulations indicates consistency.

    This check compares a number of keywords (e.g. the version of marxs) to
    indicate if the simulations where run with the same version of marxs.
    If a simulation records more information in the metadata,
    the check can be more complete.
    This function raises an `AssertionError` if the two sets of
    meta information are different.

    Parameters
    ----------
    meta1 : dict
        header from photon list 1
    meta2 : dict
        header from photon list 2
    keywords : list
        Keywords to be checked
    allow_missing : bool
        This flag allows a keyword to be missing from both dictionaries
        (e.g. MARXSGIT is not set if run with a released version); however,
        it is still an error if a key is present only in one, but not the
        dictionary.

    Raises
    ------
    AssertionError, KeyError
    '''
    for k in keywords:
        if (k in meta1) and (k in meta2):
            assert meta1[k] == meta2[k]
        else:
            if not allow_missing:
                raise KeyError(f'{k} not found in both dicts.')
            # Even with allow_missing=True
            # it's an error to have a key in only one dict
            if (k in meta1) or (k in meta2):
                raise KeyError(f'{k} found in one, but not both dicts.')


def check_energy_consistent(photons):
    '''Check if the energy is the same for all photons.

    If there is no energy colum, then this also passes.
    '''
    if 'energy' in photons.colnames:
        assert np.allclose(photons['energy'], photons['energy'][0])


class SimulationSequenceElement(MarxsElement):
    '''Base class for all elements in a simulation that processes photons.'''

    output_columns = []
    '''This is a list of strings that names the output properties.

    This gives the names of the output properties from this optical
    element.  `process_photon` or `process_photons` are responsible
    for calculating the values of these properties. For example, for a
    mirror of nested shells one might set
    ``output_columns = ['mirror_shell']``
    to pass the information on which shell the interaction took place to the
    user.

    The following properties are always included in the output and do not need
    to be listed here:

        dir : `numpy.ndarray`
            4-d direction vector of ray in homogeneous coordinates
        pos : `numpy.ndarray`
            4-d position of last interaction pf the photons with any optical
            element in homogeneous coordinates. Together with ``dir`` this
            determines the equation of the ray.
        energy : float
            Photon energy in keV.
        polarization : float
            Polarization angle of the photons.
        probability : float
            Probability that the photon continues. Set to 0 if the photon is
            absorbed, to 1 if it passes the optical element and to number
            between 0 and 1 to express a probability that the photons passes.

    '''

    id_col = None
    '''String that names an id column for output.

    Set this to a string to add an automatic numbering to the output. This is
    especially useful if there are several identical optical components that
    are used in parallel, e.g. there are four identical CCDs. Setting ``id_col
    = "CCD_ID"`` and passing an ``id_num=1, 2, 3, 4`` keyword respectively to
    each CCD will add a column ``CCD_ID`` with a value of 1,2,3, or 4 for each
    photon hitting one of those CCDs.

    Currently, this will not work with all optical elements.

    '''

    def __init__(self, **kwargs):
        self.id_num = kwargs.pop('id_num', -9)
        # We want to use id_col as a class attribute, but overwrite it if given as a kwarg
        if 'id_col' in kwargs:
            self.id_col = kwargs.pop('id_col')
        super().__init__(**kwargs)

    def add_output_cols(self, photons, colnames=[]):
        '''Add output columns to the photon array.

        This function takes the column names that are added to ``photons`` from
        several sources:

        - `id_col` (if not ``None``)
        - `output_columns`
        - the ``colnames`` parameter.

        Parameters
        ----------
        photons : `astropy.table.Table`
            Table columns are added to.
        colnames : list of elements
            Each element can be a string (in this case a float column with
            initial value ``np.nan`` is added) or a dictionay of arguments
            for `astropy.table.column.Column`. If the dictionay has a keys
            "value" then the column will be initialized to that value.
            Column names to be added; in addition several object properties can
            be used to set the column names, see description above.
        '''
        for n in self.output_columns + colnames:
            if (n is not None) and (n not in photons.colnames):
                if not isinstance(n, dict):
                    n = {'name': n, 'value': np.nan}
                val = n.pop('value', None)
                newcol = Column(length=len(photons), **n)
                if val is not None:
                    newcol[:] = val
                photons.add_column(newcol)

        # We can call this recursively, because the column will be added
        # before reaching this line agian, so we avoid infinite recursion.
        if (self.id_col is not None) and (self.id_col not in photons.colnames):
            self.add_output_cols(photons, colnames=[{'name': self.id_col,
                                                     'dtype': int,
                                                     'value': -1}])

    def __call__(self, photons, *args, **kwargs):
        return self.process_photons(photons, *args, **kwargs)


def _parse_position_keywords(kwargs):
    '''Parse keywords to define position.

    If ``pos4d`` is given, use that, otherwise look for ``position``, ``zoom``
    and ``orientation``.

    Parameters
    ----------
    pos4d : 4x4 array
        Transformation to bring an element from the default position
        (see description of individual elements) to a certain point in
        the space of homogeneous coordinates.
    position : 3-d vector in real space
        Measured from the origin of the spacecraft coordinate system.
    orientation : Rotation matrix or ``None``
        Relative orientation of the base vectors of this optical
        element relative to the orientation of the coordinate system
        of the spacecraft. The default is no rotation (i.e. the axes of both
        coordinate systems are parallel).
    zoom : float or 3-d vector
        Scale the size of an optical element in each dimension by
        ``zoom``. If ``zoom`` is a scalar, the same scale is applied
        in all dimesions. This only affects the oter dimensions of the
        optical element, not internal scales like the pixel size or
        grating constant (if defined for the optical element in
        question).  '''
    pos4d = kwargs.pop('pos4d', None)
    if pos4d is None:
        position = kwargs.pop('position', np.zeros(3))
        orientation = kwargs.pop('orientation', np.eye(3))
        zoom = kwargs.pop('zoom', 1.)
        if np.isscalar(zoom):
            zoom = np.ones(3) * zoom
        if not len(zoom) == 3:
            raise ValueError('zoom must have three elements for x,y,z or be a scalar (global zoom).')
        if np.any(np.array(zoom) <= 0):
            raise ValueError('All values in zoom must be positive-definite to keep pos4d matrix valid. Specify zero-thickness elements with display properties.')
        pos4d = affines.compose(position, orientation, zoom)
    else:
        if ('position' in kwargs) or ('orientation' in kwargs) or ('zoom' in kwargs):
            raise ValueError('If pos4d is specificed, the following keywords cannot be given at the same time: position, orientation, zoom.')

    if np.linalg.det(pos4d) == 0:
        raise ValueError('pos4d matrix is invalid (determinant is 0).')
    return pos4d
