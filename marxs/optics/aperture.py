import numpy as np
from astropy.table import Column

from .base import FlatOpticalElement
from ..base import GeometryError
from ..visualization.utils import plane_with_hole, get_color
from ..math.pluecker import h2e
from ..math.utils import anglediff
from ..simulator import Parallel

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

    @property
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

    def inner_shape(self):
        '''Return values in Eukledean space'''
        raise NotImplementedError

    def triangulate_inner_outer(self):
        '''Return a triangulation of the aperture hole embedded in a squqre.

        The size of the outer square is determined by the ``'outer_factor'`` element
        in ``self.display``.

        Returns
        -------
        xyz : np.array
            Numpy array of vertex positions in Eukeldian space
        triangles : np.array
            Array of index numbers that define triangles
        '''
        r_out = self.display.get('outer_factor', 3)
        g = self.geometry
        outer = h2e(g['center']) + r_out * np.vstack([h2e( g['v_y']) + h2e(g['v_z']),
                                                      h2e(-g['v_y']) + h2e(g['v_z']),
                                                      h2e(-g['v_y']) - h2e(g['v_z']),
                                                      h2e( g['v_y']) - h2e(g['v_z'])
        ])
        inner = self.inner_shape()
        return plane_with_hole(outer, inner)

    def _plot_mayavi(self, viewer=None):

        xyz, triangles = self.triangulate_inner_outer()

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

    def _plot_threejs(self, outfile):
        xyz, triangles = self.triangulate_inner_outer()

        from ..visualization import threejs
        materialspec = threejs.materialspec(self.display, 'MeshStandardMaterial')
        outfile.write('// APERTURE\n')
        outfile.write('var geometry = new THREE.BufferGeometry(); \n')
        outfile.write('var vertices = new Float32Array([')
        for row in xyz:
            outfile.write('{0}, {1}, {2},'.format(row[0], row[1], row[2]))
        outfile.write(''']);
        // itemSize = 3 because there are 3 values (components) per vertex
        geometry.addAttribute( 'position', new THREE.BufferAttribute( vertices, 3 ) );
        ''')
        outfile.write('var faces = new Uint16Array([')
        for row in triangles:
            outfile.write('{0}, {1}, {2}, '.format(row[0], row[1], row[2]))
        outfile.write(''']);
        // itemSize = 3 because there are 3 values (components) per triangle
        geometry.setIndex(new THREE.BufferAttribute( faces, 1 ) );
        ''')

        outfile.write('''var material = new THREE.MeshStandardMaterial({{ {materialspec} }});
        var mesh = new THREE.Mesh( geometry, material );
        scene.add( mesh );
        '''.format(materialspec=materialspec))


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

    def inner_shape(self):
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
    def __init__(self, **kwargs):
        phi = kwargs.pop('phi', [0, 2. * np.pi])
        if np.max(np.abs(phi)) > 10:
            raise ValueError('Input angles >> 2 pi. Did you use degrees (radian expected)?')
        self.phi = phi
        super(CircleAperture, self).__init__(**kwargs)

    def generate_local_xy(self, n):
        phi = np.random.uniform(self.phi[0], self.phi[1], n)
        r = np.sqrt(np.random.random(n))
        if not np.isclose(np.linalg.norm(self.geometry['v_y']),
                        np.linalg.norm(self.geometry['v_z'])):
            raise GeometryError('Aperture does not have same size in y, z direction.')

        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return x, y

    @property
    def area(self):
        '''Area covered by the aperture'''
        return 2. * np.pi * np.linalg.norm(self.geometry['v_y'])

    def inner_shape(self):
        n = self.display.get('n_inner_vertices', 90)
        phi = np.linspace(0.5 * np.pi, 2.5 * np.pi, n, endpoint=False)
        v_y = self.geometry['v_y']
        v_z = self.geometry['v_z']

        x = np.cos(phi)
        y = np.sin(phi)
        # phi could be less then full circle
        # wrap phi at lower bound (which could be negative).
        # For the default [0, 2 pi] this is a no-op
        phi = (phi - self.phi[0]) % (2 * np.pi)
        ind = phi < anglediff(self.phi)
        x[~ind] = 0
        y[~ind] = 0

        return h2e(self.geometry['center'] + x.reshape((-1, 1)) * v_y + y.reshape((-1, 1)) * v_z)
