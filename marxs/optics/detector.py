from __future__ import division
from warnings import warn

import numpy as np
from transforms3d.affines import decompose44

from ..math.pluecker import *
from .base import FlatOpticalElement


class FlatDetector(FlatOpticalElement):
    '''Flat detector with square pixels

    Processing the a photon with this detector adds four columns to the photon
    properties:

    - ``det_x``, ``det_y``: Event position on the detector in mm
    - ``detpix_x``, ``detpix_y``: Event position in detector coordinates, where
      (0, 0) is on the corner of the chip.

    The (x,y) coordinate system on the chip is such that it falls on the (y,z) plane
    of the global x,y,z coordinate system (it would be more logical to call the chip
    system (y,z), but traditionally that is not how chip coordinates are named). The
    pixel in the corner has coordinates (0, 0) in the pixel center.

    Parameters
    ----------
    pixsize : float
        size of pixels in mm
    '''
    output_columns = ['det_x', 'det_y', 'detpix_x', 'detpix_y']

    def __init__(self, pixsize=1, **kwargs):
        self.pixsize = pixsize
        super(FlatDetector, self).__init__(**kwargs)
        t, r, zoom, s = decompose44(self.pos4d)
        self.npix = [0, 0]
        self.centerpix = [0, 0]
        for i in (0, 1):
            z  = zoom[i + 1]
            self.npix[i] = 2 * z // self.pixsize
            if (2. * z / self.pixsize - self.npix[i]) > 1e-3:
                warn('Detector size is not an integer multiple of pixel size. It will be rounded.')
            self.centerpix[i] = (self.npix[i] - 1) / 2

    def process_photons(self, photons):
        intersect, h_intersect, det_coords = self.intersect(photons['dir'], photons['pos'])
        self.add_output_cols(photons)
        photons['pos'][intersect] = h_intersect[intersect]
        photons['det_x'][intersect] = det_coords[intersect, 0]
        photons['det_y'][intersect] = det_coords[intersect, 1]
        photons['detpix_x'][intersect] = det_coords[intersect, 0] / self.pixsize + self.centerpix[0]
        photons['detpix_y'][intersect] = det_coords[intersect, 1] / self.pixsize + self.centerpix[1]
        return photons
