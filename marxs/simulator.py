import numpy as np
from transforms3d.affines import decompose44

from .math.utils import translation2aff, zoom2aff, mat2aff
from .base import SimulationSequenceElement, _parse_position_keywords


class SimulationSetupError(Exception):
    pass


class Sequence(SimulationSequenceElement):
    '''A SimulationSequence is a container that summarizes several optical elements..

    Parameters
    ----------
    sequence : list
        The elements of this list are all optical elements that process photons.
    preprocess_steps : list
        The elements of this list are functions or callable objects that accept a photon list as input
        and returns no output (but changing the photon list in place, e.g. adding meta-data is
        allowed)  (*default*: ``[]``). All ``preprocess_steps`` are run before *every* optical element
        in the sequence.
        An example would be a function that writes the photon list to disk as a backup before
        every optical element or prints some informational message.
        If your function returns a modified photon list, treat it as an optical element and place it
        in `sequence`.
    postprocess_steps : list
        See `preprecess_steps` except that thee steps are run *after* each sequnece element
         (*default*: ``[]``).


    Example
    -------
    The following example shows a complete marxs simulation.
    First, we import the required modules:

    >>> from marxs import source, optics
    >>> from marxs.simulator import Sequence

    Then, we build up the parts of the simulation, source, pointing model and hardware
    of our instrument:

    >>> mysource = source.ConstantPointSource(coords=(30., 30.), flux=1e-3, energy=2.)
    >>> sky2mission = source.FixedPointing(coords=(30., 30.))
    >>> aper = optics.RectangleAperture(position=[50., 0., 0.])
    >>> mirr = optics.ThinLens(focallength=10, position=[10., 0., 0.])
    >>> ccd = optics.FlatDetector(pixsize=0.05)
    >>> sequence = [sky2mission, aper, mirr, ccd]
    >>> my_instrument = Sequence(sequence=sequence)

    Finally, we run one set of photons through the instrument:

    >>> photons_in = mysource.generate_photons(1e5)
    >>> photons_out = my_instrument(photons_in)

    Now, let us check where the photons fall on the detector:

    >>> set(photons_out['detpix_x'].round())
    set([19.0, 20.0])

    As expected, they fall right around the center of the detector (row 19 and 20 of a
    40 * 40 pixel detector).
    '''

    def __init__(self, **kwargs):
        self.sequence = kwargs.pop('sequence')
        self.preprocess_steps = kwargs.pop('preprocess_steps', [])
        self.postprocess_steps = kwargs.pop('postprocess_steps', [])
        for elem in self.sequence + self.preprocess_steps + self.postprocess_steps:
            if not callable(elem):
                raise SimulationSetupError('{0} is not callable.'.format(str(elem)))
        super(Sequence, self).__init__(**kwargs)

    def process_photons(self, photons):
        for elem in self.sequence:
            for p in self.preprocess_steps:
                p(photons)
            photons = elem(photons)
            for p in self.postprocess_steps:
                p(photons)
        return photons

class Parallel(SimulationSequenceElement):
    '''A container for several identical optical elements, e.g. chips of a CCD.

    This object describes a set of similar elements that operate in parallel,
    meaning that each photon will only interact with one of them (although that is
    not enforced by the current implementation).
    Examples for such an optical element would be the ACIS-I CCD detector on
    Chandra, that consists for four CCD chips or the HETGS gratings that are
    made up of many individual grating facets.

    The `Parallel` class requires a description of the individual components
    and a list of their positions.
    This class has build-in support to simulate uncertainties in the manufacturing
    process.
    After generation, individual positions can be adjusted by hand by
    editing the :attribute:`elem_pos`.
    Also, additional misalingments for each facetelement can be introduced by
    editing :attribute:`elem_uncertainty`. This attribute holds a list of affine
    transformation matrices.
    The global position and rotation of the combined element can be changed with
    :attribute:`uncertainty`, e.g. the represent the reproducibility of
    inserting the gratings into the beam for separate observations or the positioning
    of detectors on a detector wheel. The
    uncertainty is expressed as an affine transformation matrix.

    All uncertianty matrices should only consist of translation and rotations
    and all uncertainties should be relatively small.

    After any of the :attribute:`elem_pos`, :attribute:`elem_uncertainty` or
    :attribute:`uncertainty` is changed, :method:`generate_elements` needs to be
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
    chosen such that all rotations are done around the actual center of the
    facet or GAS respectively. "Uncertainty" roations are always done *after*
    all other rotations are accounted for.

    Use ``pos4d`` in ``elem_args`` to set a fixed rotation for all facets, e.g.
    for a CAT grating.

    Parameters
    ----------
    '''

    id_col = 'element'

    uncertainty = np.eye(4)
    '''Uncertainty of pos4d.

    The global position and rotation of the combined element can be changed with
    :attribute:`uncertainty`, e.g. the represent the reproducibility of
    inserting the gratings into the beam for separate observations or the positioning
    of detectors on a detector wheel. The
    uncertainty is expressed as an affine transformation matrix.
    '''

    elements = []
    '''List of objectselements that make up the parallel structure.

    Initially, this is an empty list, it will be filled by `generate_elements`.'''

    def __init__(self, **kwargs):

        self.elem_class = kwargs.pop('elem_class')
        # Need to operate on a copy here, to avoid changing elem_args of outer level
        self.elem_args = kwargs.pop('elem_args', {}).copy()

        elem_pos = kwargs.pop('elem_pos', None)
        if isinstance(elem_pos, dict):
            # Do some consistency checks to find the most common errors.
            keys = elem_pos.keys()
            n = len(elem_pos[keys[0]])
            # Turn dictionary of lists into list of dicts and parse position keywords
            for i in range(len(keys) -1):
                if n != len(elem_pos[keys[i+1]]):
                    raise ValueError('All elements in elem_pos must have the same number of entries.')
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
        if self.elem_pos is None:
            try:
                self.elem_pos = self.calculate_elempos()
            except NotImplementedError:
                raise ValueError('"elem_pos" must be specified as argument')
        self.elem_uncertainty = [np.eye(4)] * len(self.elem_pos)
        self.generate_elements()

    def calculate_elempos(self):
        '''Calculate the position of elements based on some algorithm.

        Classes derived from `Parallel` can overwrite this method if they want to
        provide a way to calculate the pos4d matrices for the elements that make up this
        `Parallel` element.
        This function is called in the intialization if ``elem_pos is None``.
        '''
        raise NotImplementedError

    def generate_elements(self):

        # _parse_position_keywords pops off keywords, thus operate on a copy here
        elem_args = self.elem_args.copy()
        elem_pos4d = _parse_position_keywords(elem_args)
        telem, relem, zelem, Selem = decompose44(elem_pos4d)
        if not np.allclose(Selem, 0.):
            raise ValueError('pos4 for elem includes shear, which is not supported here.')
        name = elem_args.pop('name', '')

        self.elements = []

        for i in range(len(self.elem_pos)):
            e_center, e_rot, e_zoom, stemp = decompose44(self.elem_pos[i])
            tsigelem, rsigelem, zsigelem, stemp = decompose44(self.elem_uncertainty[i])
            if not np.allclose(stemp, 0.):
                raise ElemPlacementError('Shear is not supported in the elem uncertainty.')
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

            self.elements.append(self.elem_class(pos4d = f_pos4d, name='{0}  Elem {1} in {2}'.format(name, i, self.name), id_num=i, **elem_args))


    def process_photons(self, photons):
        for elem in self.elements:
            photons = elem(photons)
        return photons

    def intersect(self, photons):
        raise NotImplementedError
