# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename

from marxs.source.labSource import LabPointSourceCone
from marxs.optics.multiLayerMirror import MultiLayerMirror
from marxs.optics.detector import FlatDetector


def test_src_mir_det():
    '''This tests that source, multilayer mirror, and detector run without producing any errors.
    '''
    string1 = get_pkg_data_filename('data/A12113.txt', package='marxs.optics')
    string2 = get_pkg_data_filename('data/ALSpolarization2.txt', package='marxs.optics')
    a = 2**(-0.5)
    rotation1 = np.array([[a, 0, -a],
                          [0, 1, 0],
                          [a, 0, a]])
    rotation2 = np.array([[0, 0, 1],
                          [0, 1, 0],
                          [-1, 0, 0]])

    source = LabPointSourceCone([10., 0., 0.], flux=100. / u.s)
    mirror = MultiLayerMirror(reflFile=string1, testedPolarization=string2,
                              position=np.array([0., 0., 0.]),
                              orientation=rotation1)
    detector = FlatDetector(1., position=np.array([0., 0., 10.]),
                            orientation=rotation2, zoom = np.array([1, 100, 100]))

    photons = source.generate_photons(100 * u.s)
    photons = mirror(photons)
    photons = detector(photons)
