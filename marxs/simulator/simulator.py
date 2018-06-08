# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from transforms3d.affines import decompose44

from ..math.utils import translation2aff, zoom2aff, mat2aff, e2h, h2e
from ..math.geometry import Geometry
from ..base import SimulationSequenceElement, _parse_position_keywords
import transforms3d
from transforms3d.utils import normalized_vector

__all__ = ['SimulationSetupError',
           'BaseContainer',
           'Sequence',
           'Parallel',
           'ParallelCalculated',
           'KeepCol',
           ]

class SimulationSetupError(Exception):
    pass


class BaseContainer(SimulationSequenceElement):
    '''Base class for containers that contain several `SimulationSequenceElement` objects.
    '''

    display = {'shape': 'container'}
    elements = []
    '''List of elements that are contained in this container.'''

    def __init__(self, **kwargs):
        self.preprocess_steps = kwargs.pop('preprocess_steps', [])
        self.postprocess_steps = kwargs.pop('postprocess_steps', [])
        for elem in self.elements + self.preprocess_steps + self.postprocess_steps:
            if not callable(elem):
                raise SimulationSetupError('{0} is not callable.'.format(str(elem)))
        super(BaseContainer, self).__init__(**kwargs)

    def __call__(self, photons):
        for elem in self.elements:
            for p in self.preprocess_steps:
                p(photons)
            photons = elem(photons)
            for p in self.postprocess_steps:
                p(photons)
        return photons

    def intersect(self, photons, **kwargs):
        raise NotImplementedError


class Sequence(BaseContainer):
    '''A `Sequence` is a container that summarizes several optical elements.

    Parameters
    ----------
    elements : list
        The elements of this list are all optical elements that process photons.
    preprocess_steps : list
        The elements of this list are functions or callable objects that accept a photon list as input
        and return no output (but changing the photon list in place, e.g. adding meta-data is
        allowed)  (*default*: ``[]``). All ``preprocess_steps`` are run before *every* optical element
        in the sequence.
        An example would be a function that writes the photon list to disk as a backup before
        every optical element or prints some informational message.
        If your function returns a modified photon list, treat it as an optical element and place it
        in `sequence`.
    postprocess_steps : list
        See ``preprocess_steps`` except that the steps are run *after* each sequence element
         (*default*: ``[]``).


    Examples
    --------
    The following example shows a complete marxs simulation.
    First, we import the required modules:

    >>> from astropy.coordinates import SkyCoord
    >>> from marxs import source, optics
    >>> from marxs.simulator import Sequence

    Then, we build up the parts of the simulation, source, pointing model and hardware
    of our instrument:

    >>> mysource = source.PointSource(coords=SkyCoord(30., 30., unit="deg"), flux=1e-3, energy=2.)
    >>> sky2mission = source.FixedPointing(coords=SkyCoord(30., 30., unit='deg'))
    >>> aper = optics.RectangleAperture(position=[50., 0., 0.])
    >>> mirr = optics.ThinLens(focallength=10, position=[10., 0., 0.])
    >>> ccd = optics.FlatDetector(pixsize=0.05)
    >>> sequence = [sky2mission, aper, mirr, ccd]
    >>> my_instrument = Sequence(elements=sequence)

    Finally, we run one set of photons through the instrument:

    >>> photons_in = mysource.generate_photons(1e5)
    >>> photons_out = my_instrument(photons_in)

    Now, let us check where the photons fall on the detector:

    >>> set(photons_out['detpix_x'].round())  # doctest: +IGNORE_OUTPUT
    {19.0, 20.0}

    As expected, they fall right around the center of the detector (row 19 and 20 of a
    40 * 40 pixel detector).
    '''

    def __init__(self, **kwargs):
        self.elements = kwargs.pop('elements')
        super(Sequence, self).__init__(**kwargs)


class Parallel(BaseContainer):
    '''A container for several identical optical elements.

    This object describes a set of similar elements that operate in
    parallel, meaning that each photon will only interact with one of
    them (although that is not enforced by the current
    implementation).  Examples for such an optical element would be
    the ACIS-I CCD detector on Chandra, that consists for four CCD
    chips or the HETGS gratings that are made up of many individual
    grating facets.

    The `Parallel` class requires a description of the individual
    components and a list of their positions.  This class has build-in
    support to simulate uncertainties in the manufacturing process.
    After generation, individual positions can be adjusted by hand by
    editing the ``elem_pos``.  Also, additional misalingments for each
    element can be introduced by editing ``elem_uncertainty``. This
    attribute holds a list of affine transformation matrices.  The
    global position and rotation of the combined element can be
    changed with `uncertainty`, e.g. the represent the reproducibility
    of inserting the gratings into the beam for separate observations
    or the positioning of detectors on a detector wheel. The
    uncertainty is expressed as an affine transformation matrix.

    All uncertainty matrices should only consist of translation and rotations
    and all uncertainties should be relatively small.

    After any of the attributes ``elem_pos``, ``elem_uncertainty`` or
    ``uncertainty`` is changed, `generate_elements` needs to be
    called to regenerate the positions of the individual elements using

    - the global position of ``Parallel.pos4d``
    - the position of each element ``Parallel.elem_pos`` relativ to the global position
    - the global uncertainty `uncertainty`.
    - the uncertainty for individual facets.

    This mechanism can be used to estimate the influence of manufacturing
    uncertainties. First, run a simulation with optimal position, then change
    the values, regenerate the facets and rerun the simulation. Comparing the
    results will allow you to estimate the effect of the manufacturing
    misalignment.

    The order in which all the transformations are applied to the facet is
    chosen such that all rotations are done around the center of the
    individual element or the whole structure respectively.
    "Uncertainty" rotations are always done *after*
    all other rotations are accounted for.

    Parameters
    ----------
    elem_class : class
        Class of the individual elements
    elem_args : dict
        Dictionary of keyword arguments that are used to initialize
        the individual arguments. This can contain position related
        keywords as listed in `pos4d`; those will be applied to *each*
        element (e.g. set the zoom for each of them).  Usually, the
        same arguments will be applied to all facets.  In the special
        case that the value of a dictionary entry is a list with the
        same length as ``elem_pos``, one value of this list will be
        used for each element. This makes it possible to specify
        e.g. different grating constants for individual facets.  (Note
        that this has to be a list. Other python types like tuple or
        np.array that behave like lists in some contexts are not
        allowed here to avoid ambiguities.)
    elem_pos : list of arrays or dictionary of lists
        Gives the position of the individual elements in the local
        coordinate system of the `Parallel` object. This can either be
        a list of (4,4) np.arrays or a dictionary with entries of
        ``pos4d`` or ``position``, ``orientation`` and ``zoom`` as
        explained in :ref:`pos4d` where each entry in the dictionary
        is a list of values ((3,3) matrices for ``orientation``, (3,)
        vectors for ``position`` etc.).  Sub-classes of `Parallel` can
        implement a method `calculate_elempos` to determine the
        position of their elements automatically. In this case, they
        should set ``elem_pos=None``.
    id_num_offset : int
        ID numbers of generated elements start at this number.

    Examples
    --------
    In this example we build up a detector made up of four CCDs. Each
    CCD is 10 mm * 10 mm large and has a pixel size of 0.01 mm. The
    CCDs are set in a square with small spaces in between.

    >>> from marxs.simulator import Parallel
    >>> from marxs.optics import FlatDetector as CCD
    >>> detect = Parallel(elem_class=CCD, elem_args={'pixsize': 0.01, 'zoom': 5},
    ...                   elem_pos={'position': [[0, -10.1, -10.1],[0, .1, -10.1],[0, -10.1, .1],[0, .1, .1]]},
    ...                   id_col='CCD_ID')

    A column that notes which CCD was hit by each photon will be added
    to the photon table when it is processed by ``photons = detect(photons)``.
    The name of this colum will be "CCD_ID".  (If
    the `id_col` argument is not passed, the name will be the generic
    "element".)

    '''

    id_col = 'element'

    uncertainty = np.eye(4)
    '''Uncertainty of pos4d.

    The global position and rotation of the combined element can be changed with
    `uncertainty`, e.g. the represent the reproducibility of
    inserting the gratings into the beam for separate observations or the positioning
    of detectors on a detector wheel. The
    uncertainty is expressed as an affine transformation matrix.
    '''

    @property
    def pos4d(self):
        return self.geometry.pos4d

    def __init__(self, **kwargs):
        geometry = kwargs.pop('geometry', Geometry)
        if isinstance(geometry, Geometry):
            self.geometry = geometry
        elif issubclass(geometry, Geometry):
            self.geometry = geometry(kwargs)

        self.id_num_offset = kwargs.pop('id_num_offset', 0)
        self.elem_class = kwargs.pop('elem_class')
        # Need to operate on a copy here, to avoid changing elem_args of outer level
        self.elem_args = kwargs.pop('elem_args', {}).copy()

        elem_pos = kwargs.pop('elem_pos')
        if isinstance(elem_pos, dict):
            # Do some consistency checks to find the most common errors.
            keys = list(elem_pos.keys())
            for i in range(len(keys) -1):
                if ( not(hasattr(elem_pos[keys[i]], '__len__')) or
                     not(hasattr(elem_pos[keys[i + 1]], '__len__')) or
                     (len(elem_pos[keys[i]]) != len(elem_pos[keys[i + 1]]))):
                    raise ValueError('All elements in elem_pos must have the same number of entries.')

            n = len(elem_pos[keys[0]])
            # Turn dictionary of lists into list of dicts and parse position keywords
            self.elem_pos = []
            for i in range(n):
                elem_pos_dict = {}
                for k in keys:
                    elem_pos_dict[k] = elem_pos[k][i]
                self.elem_pos.append(_parse_position_keywords(elem_pos_dict))
        else:
            self.elem_pos = elem_pos

        super(Parallel, self).__init__(**kwargs)

        if 'id_col' not in self.elem_args:
            self.elem_args['id_col'] = self.id_col
        self.elem_uncertainty = [np.eye(4)] * len(self.elem_pos)
        self.generate_elements()

    def move_center(self, change4d):
        '''Move the relative origin of the coordinate system.

        Individual element positions in ``elem_pos`` are relative to the
        global ``pos4d`` of this object. This function changes the
        ``pos4d`` and adjusts the ``elem_pos`` accordingly, so that the
        elements still have the same position in global coordinates.

        This function is useful when studying uncertainties. The attribute
        ``uncertainty`` can be used to apply the same change to all elements.
        Any rotation will be applied around the reference position contained
        in ``pos4d``. Use this function to change that reference position.

        Parameters
        ----------
        change4d : np.array of shape (4,4)
            Homogeneous coordinate transformation matrix
        '''
        self.geometry.pos4d = np.dot(change4d, self.pos4d)
        inv = np.linalg.inv(change4d)
        self.elem_pos = [np.dot(inv, p) for p in self.elem_pos]
        self.generate_elements()

    def generate_elements(self):
        '''Initialize all optical elements.

        After any of the ``elem_pos``, ``elem_uncertainty`` or
        ``uncertainty`` attributes has changed, `generate_elements` needs to be
        called to regenerate the positions of the individual elements using

        - the global position of ``Parallel.pos4d``
        - the position of each element ``Parallel.elem_pos`` relativ to the global position
        - the global uncertainty `uncertainty`.
        - the uncertainty for individual facets.
        '''

        self.elements = []

        for i in range(len(self.elem_pos)):
            # _parse_position_keywords pops off keywords, thus operate on a copy here
            elem_args = self.elem_args.copy()
            # check if elem_args is the same for every element
            specific_elem_args = {}
            for k, v in elem_args.items():
                if isinstance(v, list) and (len(v) == len(self.elem_pos)):
                    specific_elem_args[k] = v[i]
                else:
                    specific_elem_args[k] = v
            if 'name' not in specific_elem_args:
                specific_elem_args['name'] = 'Elem {0} in {1}'.format(i, self.name)

            elem_pos4d = _parse_position_keywords(specific_elem_args)
            telem, relem, zelem, Selem = decompose44(elem_pos4d)
            if not np.allclose(Selem, 0.):
                raise ValueError('pos4 for elem includes shear, which is not supported here.')

            e_center, e_rot, e_zoom, stemp = decompose44(self.elem_pos[i])
            tsigelem, rsigelem, zsigelem, stemp = decompose44(self.elem_uncertainty[i])
            if not np.allclose(stemp, 0.):
                raise SimulationSetupError('Shear is not supported in the elem uncertainty.')
            # Will be able to write this so much better in python 3.5,
            # but for now I don't want to nest np.dot too much so here it goes
            f_pos4d = np.eye(4)
            for m in reversed([self.pos4d,                 # global position of ParallelElement
                               self.uncertainty,           # uncertainty in global positioning
                               translation2aff(tsigelem),  # uncertaintig in translation for elem
                               translation2aff(e_center),  # translate elem center to global center
                               translation2aff(telem),     # offset for all elem. Usually 0.
                               mat2aff(rsigelem),          # uncertainty in rotation for elem
                               mat2aff(e_rot),             # Rotation of individual elem
                               mat2aff(relem),             # Rotation for all  elem, e.g. CAT gratings
                               zoom2aff(zsigelem),         # uncertainty in the zoom
                               zoom2aff(e_zoom),           # zoom of individual elem
                               zoom2aff(zelem),            # sets size for all elem
                              ]):
                assert m.shape == (4, 4)
                f_pos4d = np.dot(m, f_pos4d)
            self.elements.append(self.elem_class(pos4d = f_pos4d,
                                                 id_num=self.id_num_offset + i,
                                                 **specific_elem_args))


class ParallelCalculated(Parallel):
    '''Base class for containers that automatically determine positions of elements

    `ParallelCalculated` derives from `Parallel`. It adds a mechanism to
    automatically calculate the ``pos4d`` matrices of all elements in
    this container when the object is initialized.

    This class cannot be used directly since derived classes have to add that
    algorithm. The following text explains how that works.

    `ParallelCalculated` assumes that the ``pos4d`` matrix for each element
    can constructed from three vectors:

    - Position of the center
    - Direction of normal (x-axis of element)
    - Direction the y-axis of the element should be normal to.

    Parameters
    ----------
    pos_spec : np.array of shape (n, 4) or callable
        Specification of the center positions for all elements in homogeneous
        coordinates. If this is callable, it will be called with no argument
        and should return an np.array of shape (n, 4).
    normal_spec : np.array of shape (4, ) or callable
        Specification of the normal for each element. This can either by a point
        in homogeneous coordinates (such as the focal point of the mirror). In
        case, the normal of all elements will be directed towards that point.
        If this is a direction in homogeneous coordinates (last coordinate
        is 0), then all normal vectors are parallel to this one (e.g. to make
        CCD elements that are all perpendicular to the optical axis). Finally,
        this can be a callable. In that case, it will be called with the
        element positions from ``pos_spec`` as input and should return an array
        of the same size with normal vectors.
    parallel_spec :  np.array of shape (4, ) or callable
        Specification of a direction that one box side of each element should be
        parallel to. ``normal_spec`` defines a plane, but any rotation in that
        plane is still allowed. The element will be rotated such that one of
        the planes of the element contains the direction of ``parallel_spec``.
        If this is a homogeneous coordinate (a point or direction), see
        ``normal_spec``. If this is a callable, it will be called with two
        arguments, the output of ``pos_spec`` and ``normal_spec`` and should
        return an array of the same size.
    '''

    def __init__(self, **kwargs):
        self.pos_spec = kwargs.pop('pos_spec')
        self.normal_spec = kwargs.pop('normal_spec')
        self.parallel_spec = kwargs.pop('parallel_spec')
        kwargs['elem_pos'] = self.calculate_elempos()
        super(ParallelCalculated, self).__init__(**kwargs)

    def calculate_elempos(self):
        '''Calculate the position of elements based on some algorithm.

        Classes derived from `ParallelCalculated` can overwrite this
        method if they want to provide a way to calculate the pos4d
        matrices for the elements that make up this
        `ParallelCalculated` element.

        Returns
        -------
        pos4d : list of arrays
            List of affine transformations that bring an optical
            element centered on the origin of the coordinate system
            with the active plane in the yz-plane to the required
            facet position on the Rowland torus.

        '''
        pos4d = []

        xyzw = self.get_elemxyzw()
        normals = self.get_spec('normal_spec', xyzw)
        parallels = self.get_spec('parallel_spec', xyzw, normals)
        normals = h2e(normals)
        parallels = h2e(parallels)

        for i in range(xyzw.shape[0]):
            n = normalized_vector(normals[i, :])
            p = parallels[i, :]
            rot_mat = np.zeros((3,3))
            rot_mat[:, 0] = n
            rot_mat[:, 1] = normalized_vector(p - n * np.dot(n, p))
            rot_mat[:, 2] = normalized_vector(np.cross(n, rot_mat[:, 1]))
            pos4d.append(transforms3d.affines.compose(h2e(xyzw[i, :]), rot_mat, np.ones(3)))
        return pos4d

    def get_elemxyzw(self):
        if callable(self.pos_spec):
            return self.pos_spec()
        else:
            return self.pos_spec

    def get_spec(self, specname, xyzw, *args, **kwargs):
        spec = getattr(self, specname)
        # function
        if callable(spec):
            return spec(xyzw, *args, **kwargs)
        elif len(spec) == 4:
            # vector in homogeneous coordinates
            if spec[3] == 0:
                return np.tile(spec, (xyzw.shape[0], 1))
            # position in homogeneous coordinates
            else:
                return e2h(h2e(xyzw) - h2e(spec)[None, :], 1)
        else:
            raise ValueError(specname + 'must be callable or homogeneous coordinate.')


class KeepCol(object):
    '''Object that records the value of one column after each simulation step.

    `KeepCol` is meant to be used with the preprocess_steps or postprocess_steps
    parameters of `Sequence`. It will make a copy of one column in the photon table
    before or after each processing step and keep this copy in its `data` attribute.

    Parameters
    ----------
    colname : string
        name of column

    Attributes
    ----------
    data : list
        List of saved data. This will be empty initially.
    '''

    def __init__(self, colname):
        self.colname = colname
        self.data = []

    def __call__(self, photons):
        if self.colname not in photons.colnames:
            raise KeyError('photon list has no column {0}.'.format(self.colname))
        else:
            self.data.append(photons[self.colname].copy())

    def __array__(self):
        return np.stack(self.data)

    def format_positions(self, atol=None):
        '''Format saved position columns as a single array.

        `KeepCol` keeps the value of a column (e.g. the photon position)
        at every step of the simulation. This function reformats that list of
        columns for use graphical display programs. Note that `format_positions`
        will only work for columns that store data in homogeneous coordinates,
        for all other cases, simply call `numpy.asanyarray` on your `KeepCol`
        object.

        Parameters
        ----------
        atol : float or None
            Sometimes several consequtive elements record identical photon positions in
            `keepcol`. Those are removed from the output to speed up rendering in 3D
            programs.
            ``atol`` sets the limit up to which two positions are considered *identical*.
            See `np.allclose` for a detailed description of ``atol``.
            Set to ``None`` to skip this step.

        Returns
        -------
        pos : np.array
            Array of shape (N, n, 3), where N is the number of photons and n the number of
            unique photon positions in eukledian coordinates.
        '''
        if len(self.data) == 0:
            raise ValueError('KeepCol object contains no data.')
        if not self.data[0].ndim == 2 or self.data[0].shape[1] !=4:
            raise ValueError('format_position only works for columns that store homogeneous coordinates.')
        d = np.dstack(self.data)
        d = np.swapaxes(d, 1, 2)
        d = h2e(d)
        if atol is None:
            return d
        else:
            ind = [0]
            i = 1
            for i in range(1, d.shape[1]):
                if not np.allclose(d[:, ind[-1], :], d[:, i, :], atol=atol,
                                   equal_nan=True):
                    ind.append(i)
            return d[:, ind, :]


class Propagator(SimulationSequenceElement):
    def __init__(self, **kwargs):
        '''Move photons along the rays.

        This element moves all photons in the photon list by a specified amount
        forwards or backwards along the ray.
        Normally, photons only move forward in the simulation, but sometimes it
        can be useful to reset the photons to an earlier position, e.g. to see
        how the pattern on a detector changes for different detector positions.

        Parameters
        ----------
        distance : float
            Distance for the photons to move. Negative values move photons
            backwards.
        '''
        self.distance = kwargs.pop('distance')
        super(Propagator, self).__init__(**kwargs)

    def __call__(self, photons):
        photons['pos'] = photons['pos'] + self.distance * photons['dir']
        return photons


def propagate(photons, d):
    '''Move photons along the rays.

    This function moves all photons in the photon list by a specified amount
    forwards or backwards along the ray.
    Normally, photons only move forward in the simulation, but sometimes it
    can be useful to reset the photons to an earlier position, e.g. to see
    how the pattern on a detector changes for different detector positions.

    Parameters
    ----------
    photons : `astropy.table.Table`
        Photon table with pos and dir entries
    d : float
        Distance for the photons to move. Negative values move photons
        backwards.

    Returns
    -------
    photons : `astropy.table.Table`
        Photon table with pos and dir entries
    '''
    photons['pos'] = photons['pos'] + d * photons['dir']
    return photons
