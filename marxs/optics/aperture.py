import numpy as np
from astropy.table import Column, Table, Row

from .base import FlatOpticalElement


class BaseAperture(object):
    @staticmethod
    def add_colpos(photons):
        photoncoords = Column(name='pos', length=len(photons), shape=(4,))
        photons.add_column(photoncoords)
        photons['pos'][:, 3] = 1


class Aperture(FlatOpticalElement, BaseAperture):
    def process_photons(self, photons):
        self.add_colpos(photons)
        return photons


class SquareEntranceAperture(Aperture):
    def __init__(self, **kwargs):
        self.size = kwargs.pop('size')
        super(SquareEntranceAperture, self).__init__(**kwargs)

    def process_photons(self, photons):
        super(SquareEntranceAperture, self).process_photons(photons)
        # random positions in system of aperture
        n = len(photons)
        r1 = self.size * np.random.random(n) - 0.5 * self.size
        r2 = self.size * np.random.random(n) - 0.5 * self.size
        photons['pos'] = self.geometry['center'] + r1.reshape((-1, 1)) * self.geometry['e_y'] + r2.reshape((-1, 1)) * self.geometry['e_z']
        return photons
