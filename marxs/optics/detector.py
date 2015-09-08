from __future__ import division
import warnings

from transforms3d.affines import decompose44

from .base import FlatOpticalElement

class PixelSizeWarning(Warning):
    pass

warnings.filterwarnings("always", ".*", PixelSizeWarning)

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

    kwargs :
       see `args for optical elements`
    '''

    loc_coos_name = ['det_x', 'det_y']
    '''name for output columns that contain the interaction point in local coordinates.'''


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
                warnings.warn('Detector size is not an integer multiple of pixel size in direction {0}. It will be rounded.'.format('xy'[i]), PixelSizeWarning)
            self.centerpix[i] = (self.npix[i] - 1) / 2

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        detx = intercoos[intersect, 0] / self.pixsize + self.centerpix[0]
        dety = intercoos[intersect, 1] / self.pixsize + self.centerpix[1]
        return {'detpix_x': detx, 'detpix_y': dety}
