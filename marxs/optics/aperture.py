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
