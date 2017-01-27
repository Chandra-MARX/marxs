import numpy as np
from astropy import table
from astropy.utils.metadata import enable_merge_strategies

from .base import FlatOpticalElement
from ..base import GeometryError
from ..visualization.utils import plane_with_hole, get_color
from ..math.pluecker import h2e
from ..math.utils import anglediff
from ..simulator import BaseContainer
from .. import utils

class BaseAperture(object):
    '''Base Aperture class'''

    display = {'color': (0.0, 0.75, 0.75),
               'opacity': 0.3,
               'shape': 'plane with hole'}

    @staticmethod
    def add_colpos(photons):
        '''add columns ['pos'] to photon array'''
        photoncoords = table.Column(name='pos', length=len(photons), shape=(4,))
        photons.add_column(photoncoords)
        photons['pos'][:, 3] = 1

    @property
    def area(selv):
        '''Area of the aperture.

        This does not take into account any projection effects for
        apertures that are not perpendicular to the optical axis.
        '''
        return NotImplementedError


class FlatAperture(BaseAperture, FlatOpticalElement):
    'Base class for geometrically flat apertures defined in python'
    def generate_local_xy(selv, n):
        '''Generate x, y in the local coordinate system

        Parameters
        ----------
        n : int
            number of x, y coordinate pairs requested

        Returns
        -------
        x, y : arrays of floats
            (x, y) coordinates of the photons in the local frame
        '''
        return NotImplementedError

    def __call__(selv, photons):
        selv.add_output_cols(photons, selv.loc_coos_name)
        # Add ID number to ID col, if requested
        if selv.id_col is not None:
            photons[selv.id_col] = selv.id_num
        # Set position in different coordinate systems
        x, y = selv.generate_local_xy(len(photons))
        if selv.loc_coos_name is not None:
            photons[selv.loc_coos_name[0]] = x
            photons[selv.loc_coos_name[1]] = y
        photons['pos'] = selv.geometry('center') + x.reshape((-1, 1)) * selv.geometry('v_y') + y.reshape((-1, 1)) * selv.geometry('v_z')
        projected_area = np.dot(photons['dir'].data, - selv.geometry('e_x'))
        # Photons coming in "through the back" would have negative probabilities.
        # Unlikely to ever come up, but just in case we clip to 0.
        photons['probability'] *= np.clip(projected_area, 0, 1.)

        return photons

    def process_photons(selv, photons):
        raise NotImplementedError('You probably want to use __call__.')

    def inner_shape(selv):
        '''Return values in Eukledean space'''
        raise NotImplementedError

    def triangulate_inner_outer(selv):
        '''Return a triangulation of the aperture hole embedded in a square.

        The size of the outer square is determined by the ``'outer_factor'`` element
        in ``selv.display``.

        Returns
        -------
        xyz : np.array
            Numpy array of vertex positions in Eukeldian space
        triangles : np.array
            Array of index numbers that define triangles
        '''
        r_out = selv.display.get('outer_factor', 3)
        g = selv.geometry
        outer = h2e(g('center')) + r_out * np.vstack([h2e( g('v_y')) + h2e(g('v_z')),
                                                      h2e(-g('v_y')) + h2e(g('v_z')),
                                                      h2e(-g('v_y')) - h2e(g('v_z')),
                                                      h2e( g('v_y')) - h2e(g('v_z'))
        ])
        inner = selv.inner_shape()
        return plane_with_hole(outer, inner)


class RectangleAperture(FlatAperture):
    '''Select the position where a parallel ray from an astrophysical source starts the simulation.

    '''
    def generate_local_xy(selv, n):
        x = np.random.random(n) * 2. - 1.
        y = np.random.random(n) * 2. - 1.
        return x, y

    @property
    def area(selv):
        '''Area covered by the aperture'''
        return 4 * np.linalg.norm(selv.geometry('v_y')) * np.linalg.norm(selv.geometry('v_z'))

    def inner_shape(selv):
        g = selv.geometry
        return h2e(g('center')) + np.vstack([h2e( g('v_y')) + h2e(g('v_z')),
                                             h2e(-g('v_y')) + h2e(g('v_z')),
                                             h2e(-g('v_y')) - h2e(g('v_z')),
                                             h2e( g('v_y')) - h2e(g('v_z'))])


class CircleAperture(FlatAperture):
    '''Select the position where a parallel ray from an astrophysical source starts the simulation.

    Photons are placed in a circle. The radius of the circle is the lengths of its ``v_y`` vector.
    At this point, the aperture must have the same zoom in y and z direction; it cannot be used
    to simulate an entrance ellipse.

    Parameters
    ----------
    phi : list of 2 floats
        Bounding angles for a segment covered by the GSA. :math:`\phi=0`
        is on the positive y axis. The segment fills the space from ``phi1`` to
        ``phi2`` in the usual mathematical way (counterclockwise).
        Angles are given in radian. Note that ``phi[1] < phi[0]`` is possible if
        the segment crosses the y axis.
        (Default is the full circle.)
    '''
    def __init__(selv, **kwargs):
        phi = kwargs.pop('phi', [0, 2. * np.pi])
        if np.max(np.abs(phi)) > 10:
            raise ValueError('Input angles >> 2 pi. Did you use degrees (radian expected)?')
        selv.phi = phi
        super(CircleAperture, selv).__init__(**kwargs)

    def generate_local_xy(selv, n):
        phi = np.random.uniform(selv.phi[0], selv.phi[1], n)
        r = np.sqrt(np.random.random(n))
        if not np.isclose(np.linalg.norm(selv.geometry('v_y')),
                          np.linalg.norm(selv.geometry('v_z'))):
            raise GeometryError('Aperture does not have same size in y, z direction.')

        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return x, y

    @property
    def area(selv):
        '''Area covered by the aperture'''
        return 2. * np.pi * np.linalg.norm(selv.geometry('v_y'))

    def inner_shape(selv):
        n = selv.display.get('n_inner_vertices', 90)
        phi = np.linspace(0.5 * np.pi, 2.5 * np.pi, n, endpoint=False)
        v_y = selv.geometry('v_y')
        v_z = selv.geometry('v_z')

        x = np.cos(phi)
        y = np.sin(phi)
        # phi could be less then full circle
        # wrap phi at lower bound (which could be negative).
        # For the default [0, 2 pi] this is a no-op
        phi = (phi - selv.phi[0]) % (2 * np.pi)
        ind = phi < anglediff(selv.phi)
        x[~ind] = 0
        y[~ind] = 0

        return h2e(selv.geometry('center') + x.reshape((-1, 1)) * v_y + y.reshape((-1, 1)) * v_z)


class MultiAperture(BaseAperture, BaseContainer):
    '''Group several apertures into one class.

    Sometimes a single intrument has several physical openings where photons from an
    astrophysical source can enter, an example is XMM-Newton that operates three telescopes
    in parallel. While it is often more efficient to simulate these as entirely separate by
    running separate simulations, that is not always true.
    This class groups several apertures together.

    .. warning::

       Apertures cannot overlap. There is currently no code checking for this, but
       overlapping apertures will produce unphysical results.


    Parameters
    ----------
    elements : list
        The elements of this list are all optical elements that process photons.
    preprocess_steps : list
        The elements of this list are functions or callable objects that accept a photon list as input
        and return no output (*default*: ``[]``). All ``preprocess_steps`` are run before
        *every* aperture on just the photons that pass this aperture.
    postprocess_steps : list
        See ``preprocess_steps`` except that the steps are run *after* each aperture
         (*default*: ``[]``) on just the photons that passed that aperture.
    '''
    display = {'shape': 'container'}

    def __init__(selv, **kwargs):
        selv.elements = kwargs.pop('elements')
        selv.id_col = kwargs.pop('id_col', 'aperture')
        super(MultiAperture, selv).__init__(**kwargs)

    @property
    def area(selv):
        '''Area covered by the aperture'''
        return np.sum([e.area for e in selv.elements])

    def __call__(selv, photons):
        areas = np.array([e.area for e in selv.elements])
        aperid = np.digitize(np.random.rand(len(photons)), np.cumsum(areas) / selv.area)

        # Add ID number to ID col, if requested
        if selv.id_col is not None:
            photons[selv.id_col] = aperid
        outs = []
        for i, elem in enumerate(selv.elements):
            thisphot = photons[aperid == i]
            for p in selv.preprocess_steps:
                p(thisphot)
            thisphot = elem(thisphot)
            for p in selv.postprocess_steps:
                p(thisphot)
            outs.append(thisphot)
        with enable_merge_strategies(utils.MergeIdentical):
            photons = table.vstack(outs)

        return photons
