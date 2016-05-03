import numpy as np
from astropy.table import Column

from .base import FlatOpticalElement


class BaseAperture(object):
    @staticmethod
    def add_colpos(photons):
        photoncoords = Column(name='pos', length=len(photons), shape=(4,))
        photons.add_column(photoncoords)
        photons['pos'][:, 3] = 1


class RectangleAperture(FlatOpticalElement, BaseAperture):
    '''Select the position where a parallel ray from an astrophysical source starts the simulation.

    '''
    def process_photons(self, photons):
        self.add_colpos(photons)
        # random positions in system of aperture
        n = len(photons)
        r1 = np.random.random(n) * 2. - 1.
        r2 = np.random.random(n) * 2. - 1.
        photons['pos'] = self.geometry['center'] + r1.reshape((-1, 1)) * self.geometry['v_y'] + r2.reshape((-1, 1)) * self.geometry['v_z']
        return photons

class CircleAperture(FlatOpticalElement, BaseAperture):
    '''Select the position where a parallel ray from an astrophysical source starts the simulation.

    Photons are placed in a circle. The radius of the circle is the lengths of its ``v_y`` vector.
    At this point, the aperture must have the same zoom in y and z direction; it cannot be used
    to simulate an entrance ellipse.
    '''
    def process_photons(self, photons):
        self.add_colpos(photons)
        # random positions in system of aperture
        n = len(photons)
        phi = np.random.random(n) * 2. * np.pi
        phi = phi.reshape((-1, 1))
        r = np.sqrt(np.random.random(n))
        r = r.reshape((-1, 1))
        if not np.isclose(np.linalg.norm(self.geometry['v_y']),
                        np.linalg.norm(self.geometry['v_z'])):
            raise GeometryError('Aperture does not have same size in y, z direction.')

        photons['pos'] = self.geometry['center'] + r * (np.sin(phi) * self.geometry['v_y'] + np.cos(phi) * self.geometry['v_z'])
        return photons
