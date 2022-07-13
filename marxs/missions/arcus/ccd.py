# Licensed under GPL version 3 - see LICENSE.rst
import os
import math

import numpy as np
from astropy.table import QTable
import astropy.units as u
from scipy.stats import norm

from .utils import config

ccdfwhm = QTable.read(os.path.join(config['data']['caldb_inputdata'],
                                   'detectors',
                                   'ccd_2021', 'arcus_ccd_rmf_20210211.txt'),
                      format='ascii.no_header', names=['energy', 'FWHM'])
# Units currently not set in table
ccdfwhm['energy'] *= u.keV
ccdfwhm['FWHM'] *= u.eV


class CCDRedist():
    '''

    Notes
    -----
    For now, I'm just putting in methods one by one as I need them.
    However, in principle this class could inherit from `scipy.stat.norm`
    or maybe from a sherpa norm1d distribution or from astropy.modelling
    (which should be quantity aware already).
    Eventually, this might be a good application of a metaclass
    e.g. as in https://stackoverflow.com/questions/11349183/how-to-wrap-every-method-of-a-class
    to handles scale and loc arguments autmoatically in the way shown
    below, but for now it's easier ot just copy and paste that wrapping
    a few times as needed.


    Also, this is for instance methods. I think "norm" might use class methods
    so that's just one step more complicated...

    '''
    fwhm2sig = 2 * math.sqrt(2 * math.log(2))

    def __init__(self, tab_width=ccdfwhm):
        self.tab_width = tab_width
        if 'sigma' not in self.tab_width.colnames:
            self.tab_width['sigma'] = self.tab_width['FWHM'] / self.fwhm2sig

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
