# Licensed under GPL version 3 - see LICENSE.rst
import math
import warnings

import astropy.units as u
import numpy as np
from transforms3d.affines import decompose44
from scipy.stats import norm

from .base import FlatOpticalElement, OpticalElement
from ..utils import SimulationSetupWarning
from ..math.geometry import Cylinder


__all__ = ['FlatDetector', 'CircularDetector', 'CCDRedistNormal']


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

    ignore_pixel_warning : bool
        ignore warning if derived pixel number is not close to an integer

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

    def __init__(self, pixsize=1, ignore_pixel_warning=False, **kwargs):
        self.pixsize = pixsize
        super().__init__(**kwargs)
        t, r, zoom, s = decompose44(self.pos4d)
        self.npix = [0, 0]
        self.centerpix = [0, 0]
        for i in (0, 1):
            z  = zoom[i + 1]
            self.npix[i] = int(np.round(2. * z / self.pixsize))
            if (np.abs(2. * z / self.pixsize - self.npix[i]) > 1e-2) and not ignore_pixel_warning:
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
        The radius of the tube is given by the ``zoom`` keyword, see
        `pos4d`.  Use ``zoom[0] == zoom[1]`` to make a circular
        tube. ``zoom[0] != zoom[1]`` gives an elliptical
        profile. ``zoom[2]`` sets the extension in the z direction.
    pixsize : float size of pixels in mm

    '''
    loc_coos_name = ['det_phi', 'det_y']

    detpix_name = ['detpix_x', 'detpix_y']
    '''name for output columns that contain this pixel number.'''

    display = {'color': (1.0, 1.0, 0.),
               'opacity': 0.7,
               }

    centerpix = [0, 0]

    default_geometry = Cylinder

    def __init__(self, pixsize=1, **kwargs):
        self.pixsize = pixsize
        super().__init__(**kwargs)

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        detx = intercoos[intersect, 0] * self.geometry['R'] / self.pixsize + self.centerpix[0]
        dety = intercoos[intersect, 1] / self.pixsize + self.centerpix[1]
        return {self.detpix_name[0]: detx, self.detpix_name[1]: dety}



class CCDRedistNormal(FlatOpticalElement):
    """Redistribute energies according to a normal distribution

    No detector has infinite energy resolution. This class redistributes the
    the energy according to a normal distribution.

    This class serves a dual role. It can be used as an optical element within an
    instrument to cheaply simulate a CCD response (not including effects like frame
    transfer or CTI), but it can also be used in the OSIP generators in
    `marxs.reduction.osip`.

    Parameters
    ----------
    tab_width : astropy.table.QTable
        Table with columns "energy" and either "sigma" or "FWHM"

    Notes
    -----
    EXPERIMENTAL.
    For now, I'm just putting in methods one by one as I need them to help
    me develop how the stable API should look like.
    In principle this class could inherit from `scipy.stat.norm`
    or maybe from a sherpa norm1d distribution or from astropy.modelling
    (which should be quantity aware already).
    Eventually, this might be a good application of a metaclass
    e.g. as in https://stackoverflow.com/questions/11349183/how-to-wrap-every-method-of-a-class
    to handles scale and loc arguments automatically in the way shown
    below, but for now it's easier ot just copy and paste that wrapping
    a few times as needed.


    Also, this is for instance methods. I think "norm" might use class methods
    so that's just one step more complicated...

    """

    fwhm2sig = 2 * math.sqrt(2 * math.log(2))

    energycol = "energy_detected"

    def __init__(self, **kwargs):
        self.tab_width = kwargs.pop("tab_width")
        if 'sigma' not in self.tab_width.colnames:
            self.tab_width['sigma'] = self.tab_width['FWHM'] / self.fwhm2sig
        super().__init__(**kwargs)

    @u.quantity_input(energy=u.keV, equivalencies=u.spectral())
    def sig_ccd(self, energy):
        '''Return the Gaussian sigma of the width of the CCD resolution

        Parameters
        ----------
        energy : `~astropy.units.quantity.Quantity`
            True photon energy.

        Returns
        -------
        sigma : `~astropy.units.quantity.Quantity`
            Width of the Gaussian
        '''
        return np.interp(energy.to(u.keV, equivalencies=u.spectral()),
                         self.tab_width['energy'],
                         self.tab_width['sigma'])

    @u.quantity_input(x=u.keV, loc=u.keV, equivalencies=u.spectral())
    def cdf(self, x, loc):
        scale = self.sig_ccd(loc)
        return norm().cdf(((x - loc) / scale).decompose())

    @u.quantity_input(loc=u.keV, equivalencies=u.spectral())
    def interval(self, alpha, loc):
        scale = self.sig_ccd(loc)
        return np.broadcast_to(norm.interval(alpha), (len(loc), 2)).T * scale + loc

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        # Implicit unit of photon energy is keV, but here we need it written out.
        phot_en = u.Quantity(photons["energy"])
        det_en = norm.rvs(
            loc=phot_en,
            scale=self.sig_ccd(phot_en).to(u.keV, equivalencies=u.spectral()),
        )
        return {self.energycol: det_en}
