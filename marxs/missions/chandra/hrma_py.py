'''A pure-python approximation of the Chandra mirror.

This approximation is crude, but it's useful to have for running
examples in environments where  `classic MARX`_ is not installed.
'''
import numpy as np

from .data import NOMINAL_FOCALLENGTH
from ...import optics

mirror_radii = np.array([[598.,610.],[481, 491], [424, 433], [315,322]])

class Aperture(optics.MultiAperture):
    '''Chandra openign aperture of four rings above the mirror shells.
    '''
    def __init__(self, **kwargs):
        if not 'id_col' in kwargs:
            kwargs['id_col'] = 'mirror_shell'
        if not 'elements' in kwargs:
            kwargs['elements'] = [optics.CircleAperture(position=[NOMINAL_FOCALLENGTH, 0, 0],
                              zoom=[1, r[1], r[1]], r_inner=r[0]) for r in mirror_radii]
        super(Aperture, self).__init__(**kwargs)


class HRMA(optics.FlatStack):
    '''High-resolution mirror assembly in a pure Python implementation

    This class delivers a rough approximation (a few percent for on-axis sources)
    to the Chandra mirror. Many important effects are missing (e.g. the mirror
    reflectivity in this module is not energy dependent, neither is the mirror scatter).
    Use `marxs.optics.MarxMirror` if possible.
    '''
    def __init__(self, **kwargs):
        'asf'
        kwargs['position'] = [NOMINAL_FOCALLENGTH, 0, 0]
        kwargs['zoom'] = [1, 650., 650]
        kwargs['elements'] = [optics.PerfectLens,
                              optics.RadialMirrorScatter,
                              optics.EnergyFilter]
        kwargs['keywords'] = [{'focallength': NOMINAL_FOCALLENGTH},
                              {'inplanescatter': 3.6e-6,
                               'perpplanescatter': 1.2e-6},
                              {'filterfunc': lambda x: np.ones_like(x) * 0.66,
                               'name': 'support spider, reflectivity, et al.'}]
        super(HRMA, self).__init__(**kwargs)
