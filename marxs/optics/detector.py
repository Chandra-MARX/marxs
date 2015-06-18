import numpy as np
from astropy.table import Column

from ..math.pluecker import *
from .base import FlatOpticalElement


class InfiniteFlatDetector(FlatOpticalElement):
    '''Infinitely extended flat detector with square pixels

    Pixel Size is 1, use a scaling on pos4d to change that.

    Output columns are names det_y, det_z not x,y.
    Should I change that?
    '''
    def process_photons(self, photons):
        intersect, h_intersect, temp = self.intersect(photons['dir'], photons['pos'])
        photons['pos'][intersect] = h_intersect[intersect]
        det_pos = h2e(h_intersect) - h2e(self.geometry['center'])
        photons.add_column(Column(np.dot(det_pos, h2e(self.geometry['e_y'])), name='det_y'))
        photons.add_column(Column(np.dot(det_pos, h2e(self.geometry['e_z'])), name='det_z'))
        return photons
