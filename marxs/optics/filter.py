'''This module contains filters, e.g. an optical blocking filter or CCD contamination.
'''
import numpy as np

from .base import FlatOpticalElement


class EnergyFilter(FlatOpticalElement):
    '''Energy dependent filter.

    Parameters
    ----------
    filterfunc : callable
        A function that calculates the probability for each photon to pass
        through the filter based on the photon energy in keV. The function
        signature should be ``p = func(en)``, where ``p, en`` are 1-d arrays
        of floats with the same number of elements.

    Example
    -------
    >>> from scipy.interpolate import interp1d
    >>> from marxs.optics import EnergyFilter
    >>> energygrid = [.1, .5, 1., 2., 5.]
    >>> filtercurve = [.1, .5, .9, .9, .5]
    >>> f = interp1d(energygrid, filtercurve)
    >>> blockingfilter = EnergyFilter(filterfunc=f)
    '''
    def __init__(self, **kwargs):
        self.filterfunc = kwargs.pop('filterfunc')
        super(EnergyFilter, self).__init__(**kwargs)

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        p =  self.filterfunc(photons['energy'][intersect])
        if np.any(p < 0.) or np.any(p > 1.):
            raise ValueError('Probabilities returned by filterfunc must be in interval [0, 1].')
        return {'probability': p}
