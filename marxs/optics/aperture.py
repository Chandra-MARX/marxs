# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from astropy import table
import astropy.units as u

from .base import FlatOpticalElement
from ..base import GeometryError
from ..math.geometry import RectangleHole, CircularHole
from ..simulator import BaseContainer


class BaseAperture(object):
    '''Base Aperture class'''

    display = {'color': (0.0, 0.75, 0.75),
               'opacity': 0.3,
               'shape': 'triangulation'}

    @staticmethod
    def add_colpos(photons):
        '''add columns ['pos'] to photon array'''
        if 'pos' not in photons.colnames:
            photoncoords = table.Column(name='pos', length=len(photons),
                                        shape=(4,))
            photons.add_column(photoncoords)
            photons['pos'][:, 3] = 1
            photons['pos'].unit = u.mm

    @property
    def area(self):
        '''Area of the aperture.

        This does not take into account any projection effects for
        apertures that are not perpendicular to the optical axis.
        '''
        return NotImplementedError


class FlatAperture(BaseAperture, FlatOpticalElement):
    'Base class for geometrically flat apertures defined in python'

    def __call__(self, photons):
        # The last two arguments make no sense for apertures - the
        # intercoos and interpoos are assigned in specific_process_photons
        # instead of derived from intersect - but we need something here
        # to keep the interface the same as in FlatOpticalElement.
        self.add_colpos(photons)
        self.process_photons(photons, np.ones(len(photons), dtype=bool),
                             photons['pos'],
                             np.zeros((len(photons), 2)))
        return photons

    def generate_local_xy(self, n):
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

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        x, y = self.generate_local_xy(intersect.sum())

        intercoos[intersect, 0] = x
        intercoos[intersect, 1] = y
        interpos[intersect, :] = self.geometry['center'] + x.reshape((-1, 1)) * self.geometry['v_y'] + y.reshape((-1, 1)) * self.geometry['v_z']
        projected_area = np.dot(photons['dir'][intersect].data,
                                - self.geometry['e_x'])
        # Photons coming "through the back" would have negative probabilities.
        # Unlikely to ever come up, but just in case we clip to 0.
        return {'probability': np.clip(projected_area, 0, 1.)}

    def outer_shape(self):
        '''Return values in Eukledian space.'''
        return self.xyz_square(self.display.get('outer_factor', 3))


class RectangleAperture(FlatAperture):
    '''Select the position where a parallel ray from an astrophysical source starts the simulation.

    '''

    default_geometry = RectangleHole

    def generate_local_xy(self, n):
        x = np.random.random(n) * 2. - 1.
        y = np.random.random(n) * 2. - 1.
        return x, y

    @property
    def area(self):
        '''Area covered by the aperture'''
        return 4 * np.linalg.norm(self.geometry['v_y']) * np.linalg.norm(self.geometry['v_z']) * u.mm**2


class CircleAperture(FlatAperture):
    '''Select the position where a parallel ray from an astrophysical source starts the simulation.

    Photons are placed in a circle. The radius of the circle is the lengths of
    its ``v_y`` vector.  At this point, the aperture must have the same zoom in
    y and z direction; it cannot be used to simulate an entrance ellipse.

    Parameters
    ----------
    phi : list of 2 floats
        Bounding angles for a segment covered by the GSA. :math:`\phi=0`
        is on the positive y axis. The segment fills the space from ``phi1`` to
        ``phi2`` in the usual mathematical way (counterclockwise).
        Angles are given in radian. Use ``phi[1] < 0`` if the segment crosses
        the y axis. (Default is the full circle.)
    r_inner : float
        Inner radius for ring-like apertures. Default is 0 (full circle). If
        `r_inner` is non-zero, plotting the aperture will fill in the inner
        region.  If this is not desired because several `CircleApertures` are
        stacked into each other, ``self.display['inner_factor']`` can be used
        to restrict the radius range where the inner disk is displayed in a
        plot.
    '''

    default_geometry = CircularHole

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.geometry['r_inner'] > np.linalg.norm(self.geometry['v_y']):
            raise ValueError('r_inner must be less than size of full aperture.')

        if not np.isclose(np.linalg.norm(self.geometry['v_y']),
                          np.linalg.norm(self.geometry['v_z'])):
            raise GeometryError('Aperture does not have the same size in y, z direction.')

    def generate_local_xy(self, n):
        phi = np.random.uniform(self.geometry.phi[0], self.geometry.phi[1], n)
        # normalize r_inner
        r_inner = self.geometry['r_inner'] / np.linalg.norm(self.geometry['v_y'])
        r = np.sqrt(np.random.uniform(r_inner**2, 1., n))

        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return x, y

    @property
    def area(self):
        '''Area covered by the aperture'''
        A_circ = np.pi * (np.linalg.norm(self.geometry['v_y'])**2 - self.geometry['r_inner']**2)
        return (self.geometry.phi[1] - self.geometry.phi[0])  / (2 * np.pi) * A_circ * u.mm**2


class MultiAperture(BaseAperture, BaseContainer):
    '''Group several apertures into one class.

    Sometimes a single intrument has several physical openings where photons
    from an astrophysical source can enter, an example is XMM-Newton that
    operates three telescopes in parallel. While it is often more efficient to
    simulate these as entirely separate by running separate simulations, that
    is not always true. This class groups several apertures together.

    .. warning::

       Apertures cannot overlap. There is currently no code checking for this,
       but overlapping apertures will produce unphysical results.


    Parameters
    ----------
    elements : list
        The elements of this list are all optical elements that process photons.
    preprocess_steps : list
        The elements of this list are functions or callable objects that accept
        a photon list as input and return no output (*default*: ``[]``). All
        ``preprocess_steps`` are run before *every* aperture on just the
        photons that pass this aperture.
    postprocess_steps : list
        See ``preprocess_steps`` except that the steps are run *after* each
        aperture (*default*: ``[]``) on just the photons that passed that
        aperture.

    '''
    display = {'shape': 'container'}

    def __init__(self, **kwargs):
        self.elements = kwargs.pop('elements')
        self.id_col = kwargs.pop('id_col', 'aperture')
        self.id_num_offset = kwargs.pop('id_num_offset', 0)
        super().__init__(**kwargs)
        for i, elem in enumerate(self.elements):
            elem.id_col = self.id_col
            elem.id_num = self.id_num_offset + i

    @property
    def area(self):
        '''Area covered by the aperture'''
        return u.Quantity([e.area for e in self.elements]).sum()

    def __call__(self, photons):
        self.add_colpos(photons)
        areas = u.Quantity([e.area for e in self.elements])
        aperid = np.digitize(np.random.rand(len(photons)), np.cumsum(areas) / self.area)
        np.random.shuffle(aperid)

        for i, elem in enumerate(self.elements):
            for p in self.preprocess_steps:
                p(photons)
            # The following line differs from "normal" optical
            # elements. In other elements "intersect" decides
            # which photons are touched, here we pass that in.
            photons = elem.process_photons(photons, aperid==i,
                                           photons['pos'],
                                           np.zeros((len(photons), 2)))
            for p in self.postprocess_steps:
                p(photons)
        return photons
