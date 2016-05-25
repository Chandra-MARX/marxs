import numpy as np
from astropy.table import Column

from .base import FlatOpticalElement
from ..base import GeometryError
from ..visualization.utils import plane_with_hole, get_color
from ..math.pluecker import h2e

class BaseAperture(object):
    '''Base Aperture class'''

    display = {'color': (0.0, 0.75, 0.75),
               'opacity': 0.3}

    @staticmethod
    def add_colpos(photons):
        '''add columns ['pos'] to photon array'''
        photoncoords = Column(name='pos', length=len(photons), shape=(4,))
        photons.add_column(photoncoords)
        photons['pos'][:, 3] = 1

    def area(self):
        '''Area of the aperture.

        This does not take into account any projection effects for
        apertures that are not perpendicular to the optical axis.
        '''
        return NotImplementedError


class FlatAperture(BaseAperture, FlatOpticalElement):
    'Base class for geometrically flat apertures defined in python'
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

    def process_photons(self, photons):
        self.add_output_cols(photons, self.loc_coos_name)
        # Add ID number to ID col, if requested
        if self.id_col is not None:
            photons[self.id_col] = self.id_num
        # Set position in different coordinate systems
        x, y = self.generate_local_xy(len(photons))
        if self.loc_coos_name is not None:
            photons[self.loc_coos_name[0]] = x
            photons[self.loc_coos_name[1]] = y
        photons['pos'] = self.geometry['center'] + x.reshape((-1, 1)) * self.geometry['v_y'] + y.reshape((-1, 1)) * self.geometry['v_z']

        return photons

    def _plot_mayavi_inner_shape(self):
        '''Return values in Eukledean space'''
        raise NotImplementedError

    def _plot_mayavi(self, viewer=None):

        r_out = self.display.get('outer_factor', 3)
        g = self.geometry
        outer = h2e(g['center']) + r_out * np.vstack([h2e( g['v_y']) + h2e(g['v_z']),
                                                      h2e(-g['v_y']) + h2e(g['v_z']),
                                                      h2e(-g['v_y']) - h2e(g['v_z']),
                                                      h2e( g['v_y']) - h2e(g['v_z'])
        ])
        inner = self._plot_mayavi_inner_shape()
        xyz, triangles = plane_with_hole(outer, inner)

        from mayavi.mlab import triangular_mesh

        # turn into valid color tuple
        self.display['color'] = get_color(self.display)
        t = triangular_mesh(xyz[:, 0], xyz[:, 1], xyz[:, 2], triangles, color=self.display['color'])
        # No safety net here like for color converting to a tuple.
        # If the advanced properties are set you are on your own.
        prop = t.module_manager.children[0].actor.property
        for n in prop.trait_names():
            if n in self.display:
                setattr(prop, n, self.display[n])
        return t


class RectangleAperture(FlatAperture):
    '''Select the position where a parallel ray from an astrophysical source starts the simulation.

    '''
    def generate_local_xy(self, n):
        x = np.random.random(n) * 2. - 1.
        y = np.random.random(n) * 2. - 1.
        return x, y

    @property
    def area(self):
        '''Area covered by the aperture'''
        return np.linalg.norm(self.geometry['v_y']) * np.linalg.norm(self.geometry['v_z'])

    def _plot_mayavi_inner_shape(self):
        g = self.geometry
        return h2e(g['center']) + np.vstack([h2e( g['v_y']) + h2e(g['v_z']),
                                             h2e(-g['v_y']) + h2e(g['v_z']),
                                             h2e(-g['v_y']) - h2e(g['v_z']),
                                             h2e( g['v_y']) - h2e(g['v_z'])])


class CircleAperture(FlatAperture):
    '''Select the position where a parallel ray from an astrophysical source starts the simulation.

    Photons are placed in a circle. The radius of the circle is the lengths of its ``v_y`` vector.
    At this point, the aperture must have the same zoom in y and z direction; it cannot be used
    to simulate an entrance ellipse.


    '''
    def generate_local_xy(self, n):
        phi = np.random.random(n) * 2. * np.pi
        r = np.sqrt(np.random.random(n))
        if not np.isclose(np.linalg.norm(self.geometry['v_y']),
                        np.linalg.norm(self.geometry['v_z'])):
            raise GeometryError('Aperture does not have same size in y, z direction.')

        x = r * np.sin(phi)
        y = r * np.cos(phi)
        return x, y

    @property
    def area(self):
        '''Area covered by the aperture'''
        return 2. * np.pi * np.linalg.norm(self.geometry['v_y'])

    def _plot_mayavi_inner_shape(self):
        n = self.display.get('n_inner_vertices', 90)
        phi = np.linspace(0.5 * np.pi, 2.5 * np.pi, n, endpoint=False)
        v_y = self.geometry['v_y']
        v_z = self.geometry['v_z']

        x = np.cos(phi)
        y = np.sin(phi)

        return h2e(self.geometry['center'] + x.reshape((-1, 1)) * v_y + y.reshape((-1, 1)) * v_z)
