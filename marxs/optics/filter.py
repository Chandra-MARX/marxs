'''This module contains filters, e.g. an optical blocking filter or CCD contamination.
'''
import numpy as np

from .base import OpticalElement, FlatOpticalElement


class GlobalEnergyFilter(OpticalElement):
    '''Energy dependent filter that globally affects all photons.

    This element is used on all photons in the list, there is no geometrical
    position associated with it. Consequently, there is no update of the position
    or direction for each photon.
    Use this element for global filters, that are not directly associated with any particular
    physical object, e.g. to apply a energy based mirror efficiency after
    passing the photons through one of the perfect efficiency mirror models.

    Parameters
    ----------
    filterfunc : callable
        A function that calculates the probability for each photon to pass
        through the filter based on the photon energy in keV. The function
        signature should be ``p = func(en)``, where ``p, en`` are 1-d arrays
        of floats with the same number of elements.

    Examples
    --------
    >>> from scipy.interpolate import interp1d
    >>> from marxs.optics import GlobalEnergyFilter
    >>> energygrid = [.1, .5, 1., 2., 5.]
    >>> filtercurve = [.1, .5, .9, .9, .5]
    >>> f = interp1d(energygrid, filtercurve)
    >>> blockingfilter = GlobalEnergyFilter(filterfunc=f)

    See Also
    --------
    marxs.optics.filter.EnergyFilter
    '''
    def __init__(self, **kwargs):
        self.filterfunc = kwargs.pop('filterfunc')
        super(GlobalEnergyFilter, self).__init__(**kwargs)

    def __call__(self, photons):
        p =  self.filterfunc(photons['energy'])
        if np.any(p < 0.) or np.any(p > 1.):
            raise ValueError('Probabilities returned by filterfunc must be in interval [0, 1].')
        photons['probability'] *= p
        return photons


class EnergyFilter(FlatOpticalElement):
    '''Energy dependent filter with position, size etc.

    Parameters
    ----------
    filterfunc : callable
        A function that calculates the probability for each photon to pass
        through the filter based on the photon energy in keV. The function
        signature should be ``p = func(en)``, where ``p, en`` are 1-d arrays
        of floats with the same number of elements.

    Examples
    --------
    >>> from scipy.interpolate import interp1d
    >>> from marxs.optics import EnergyFilter
    >>> energygrid = [.1, .5, 1., 2., 5.]
    >>> filtercurve = [.1, .5, .9, .9, .5]
    >>> f = interp1d(energygrid, filtercurve)
    >>> blockingfilter = EnergyFilter(filterfunc=f, position=[4, 1, 0], zoom=4)

    See Also
    --------
    marxs.optics.filter.GlobalEnergyFilter
    '''

    display = {'color': (1.0, 0., 0.),
               'opacity': 0.5,
    }
    def __init__(self, **kwargs):
        self.filterfunc = kwargs.pop('filterfunc')
        super(EnergyFilter, self).__init__(**kwargs)

    def specific_process_photons(self, photons, intersect, interpos, intercoos):
        p =  self.filterfunc(photons['energy'][intersect])
        if np.any(p < 0.) or np.any(p > 1.):
            raise ValueError('Probabilities returned by filterfunc must be in interval [0, 1].')
        return {'probability': p}
