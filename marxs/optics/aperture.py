import numpy as np
from astropy.table import Column

from .base import FlatOpticalElement
from ..base import GeometryError


class BaseAperture(object):
    '''Base Aperture class'''

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
