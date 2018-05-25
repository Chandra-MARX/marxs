# Licensed under GPL version 3 - see LICENSE.rst
from __future__ import division
import warnings

import numpy as np
from transforms3d.affines import decompose44, compose

from .base import FlatOpticalElement, OpticalElement
from ..math.utils import h2e
from ..utils import SimulationSetupWarning
from ..math.geometry import Cylinder

class FlatDetector(FlatOpticalElement):
    '''Flat detector with square pixels

    Processing the a photon with this detector adds four columns to the photon
    properties:

    - ``det_x``, ``det_y``: Event position on the detector in mm
    - ``detpix_x``, ``detpix_y``: Event position in detector coordinates, where
      (0, 0) is on the corner of the chip.

    The (x,y) coordinate system on the chip is such that it falls on the (y,z)
    plane of the global x,y,z coordinate system (it would be more logical to
    call the chip system (y,z), but traditionally that is not how chip
    coordinates are named). The pixel in the corner has coordinates (0, 0) in
    the pixel center.

    Parameters
    ----------
    pixsize : float
        size of pixels in mm

    kwargs :
       see `args for optical elements`

    '''

    loc_coos_name = ['det_x', 'det_y']
    '''name for output columns that contain the interaction point in local coordinates.'''

    detpix_name = ['detpix_x', 'detpix_y']
    '''name for output columns that contain this pixel number.'''



    display = {'color': (1.0, 1.0, 0.),
               'shape': 'box',
               'box-half': '+x'}

    def __init__(self, pixsize=1, **kwargs):
        self.pixsize = pixsize
        super(FlatDetector, self).__init__(**kwargs)
        t, r, zoom, s = decompose44(self.pos4d)
        self.npix = [0, 0]
        self.centerpix = [0, 0]
        for i in (0, 1):
            z  = zoom[i + 1]
            self.npix[i] = int(np.round(2. * z / self.pixsize))
            if np.abs(2. * z / self.pixsize - self.npix[i]) > 1e-2:
                warnings.warn('Detector size is not an integer multiple of pixel size in direction {0}. It will be rounded.'.format('xy'[i]), SimulationSetupWarning)
            self.centerpix[i] = (self.npix[i] - 1) / 2

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        detx = intercoos[intersect, 0] / self.pixsize + self.centerpix[0]
        dety = intercoos[intersect, 1] / self.pixsize + self.centerpix[1]
        return {self.detpix_name[0]: detx, self.detpix_name[1]: dety}

class CircularDetector(OpticalElement):
    '''A detector shaped like a ring or tube.

    This detector is shaped like a tube. The form is a circle in the xy plane
    and and flat along the z-direction.  While most CCDs are flat in practice,
    the `CircularDetector` simulates a setup that can follow the Rowland circle
    geometry exactly which is useful, e.g. to study the resolution of a
    spectrograph without worrying about the details of the detector geometry.

    Parameters
    ----------
    position, orientation, zoom, pos4d : see description of `pos4d`
        The radius of the tube is given by the ``zoom`` keyword, see `pos4d`.
        Use ``zoom[0] == zoom[1]`` to make a circular tube. ``zoom[0] != zoom[1]`` gives
        an elliptical profile. ``zoom[2]`` sets the extension in the z direction.
    pixsize : float
        size of pixels in mm
    phi_offset : float
        This defines the center (pixel = 0) in radian. In other words:
        The output ``det_phi`` column has its zero position at ``phi_offset`` and
        values for ``phi`` are in the range [-pi, pi].

    '''
    loc_coos_name = ['det_phi', 'det_y']

    detpix_name = ['detpix_x', 'detpix_y']
    '''name for output columns that contain this pixel number.'''

    display = {'color': (1.0, 1.0, 0.),
               'opacity': 0.7,
               }

    centerpix = [0, 0]

    def __init__(self, pixsize=1, **kwargs):
        self.pixsize = pixsize
        if 'geometry' not in kwargs:
            kwargs['geometry'] = Cylinder
        super(CircularDetector, self).__init__(**kwargs)

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        detx = intercoos[intersect, 0] * self.geometry['R'] / self.pixsize + self.centerpix[0]
        dety = intercoos[intersect, 1] / self.pixsize + self.centerpix[1]
        return {self.detpix_name[0]: detx, self.detpix_name[1]: dety}
