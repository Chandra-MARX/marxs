# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from astropy.table import Table
from astropy.io import ascii
from astropy.utils.data import get_pkg_data_filename

from transforms3d import euler
from ...math.utils import norm_vector
from ..multiLayerMirror import MultiLayerMirror, FlatBrewsterMirror

import pytest

def test_photon_reflection():
    ''' tests that three photons are accurately reflected

    TODO: This test should be broken down into smaller components, as should the
    multilayer mirror process photons function.
    '''
    pos = np.tile([1.,0.,0.,1.], (4,1))
    # hitting y = 23mm, 26mm, 24mm; std dev = 0.02, 0.04, 0.01; max = 5.81, 0.42, 6.21; lambda = 3, 6, 4
    dir = np.array([[-1., -1.5, 0., 0],
                    [-1., 1.5, 0., 0],
                    [-1., -0.5, 13., 0],
                    [-1., -1.5, 0., 0]])
    # note: these photons will not hit at 45 degrees (only ok for testing purposes)
    polarization = np.array([[0., 0., 1., 0],
                             [1., 0., 0., 0],
                             [1., 0., 0., 0],
                             [1. / np.sqrt(2.), 0., 1. / np.sqrt(2.), 0.]])
    # z axis is the parallel polarization, v_1, so crossing that with direction (perpendicular to z) is v_2
    polarization[1, 0:3] = np.cross(dir[1,0:3], polarization[0,0:3])
    polarization[1, 0:3] /= np.linalg.norm(polarization[1,0:3])

    photons = Table({'pos': pos,
                     'dir': dir,
                     'energy': [1.23984282 / 3.02, 1.23984282 / 6, 0.4,
                                1.23984282 / 3.02],
                     'polarization': polarization,
                     'probability': [1., 1., 1., 1.]})
    mirror = MultiLayerMirror(reflFile=get_pkg_data_filename('data/testFile_mirror.txt', package='marxs.optics'),
                              testedPolarization=get_pkg_data_filename('data/ALSpolarization2.txt',
                                                                       package='marxs.optics'),
                              zoom=[1, 24.5, 12])
    photons = mirror(photons)

    # confirm reflection angle
    expected_dir = norm_vector(np.array([[1., -1.5, 0., 0],
                                         [1., 1.5, 0., 0],
                                         [-1., -0.5, 13., 0],
                                         [1., -1.5, 0., 0]]))
    assert np.allclose(norm_vector(np.array(photons['dir'])), expected_dir)

    # confirm reflection probability
    polarizedFile = ascii.read(get_pkg_data_filename('data/ALSpolarization2.txt', package='marxs.optics'))
    correctTestPolar = np.interp(1.23984282 / 3.02,
                                 polarizedFile['Photon energy'] / 1000,
                                 polarizedFile['Polarization'])
    expected_prob = np.array([0.0581 * np.exp(-0.5) * 1. / correctTestPolar,
                              0.0042 * 0.,
                              1.,
                              0.5 * 0.0581 * np.exp(-0.5) * 1. / correctTestPolar])
    assert np.allclose(photons['probability'], expected_prob)

    # confirm correct new polarization
    expected_pol = np.array([0., 0, 1., 0])
    simulated_pol = norm_vector(photons['polarization'])
    for i in [0, 3]:
        assert (np.allclose(simulated_pol[i], expected_pol) or
                np.allclose(simulated_pol[i], -expected_pol))
    # prob of relfection is 0, so pol direction is undefined.
    assert np.all(np.isnan(simulated_pol[1][:3]))


def test_photon_reflection2():
    '''more rigorous reflection test'''
    pos = np.array([[0., 0., 1., 1],
                    [0., -0.6, 1., 1],
                    [0., 0., 1., 1],
                    [0., 0.5, 1., 1]])
    # hitting y = 24.5mm, 24mm, 24.5mm, 25mm;
    # ---XXX--- std dev = 0.02, 0.04, 0.01; max = 5.81, 0.42, 6.21; lambda = 3, 6, 4
    dir = np.array([[0., 0., -1., 0],
                    [0., 0.1, -1., 0],
                    [0., 0., -1., 0],
                    [0., 0., -1., 0]])
    # note: these photons will not hit at 45 degrees (only ok for testing purposes)
    polarization = np.array([[0., 1., 0., 0],
                             [0., 1., 0.1, 0],
                             [1., 0., 0., 0],
                             [1., 0., 0., 0]])
    # z axis is the parallel polarization, v_1, so crossing that with direction (perpendicular to z) is v_2

    photons = Table({'pos': pos, 'dir': dir,
                     'energy': [1.23984282 / 3.02, 1.23984282 / 6, 0.4, 0.5],
                     'polarization': polarization,
                     'probability': [1., 1., 1., 1.]})
    a = 2**(-0.5)
    mirror = MultiLayerMirror(reflFile=get_pkg_data_filename('data/testFile_mirror.txt', package='marxs.optics'),
                              testedPolarization=get_pkg_data_filename('data/ALSpolarization2.txt',
                                                                       package='marxs.optics'),
                              orientation=np.array([[-a, 0, a],[0, -1, 0],[a, 0, a]]))
    photons = mirror(photons)


    # confirm reflection angle
    expected_dir = norm_vector(np.array([[-1., 0., 0., 0], [-1., 0.1, 0., 0], [-1., 0., 0., 0], [-1., 0., 0., 0]]))
    assert np.allclose(norm_vector(np.array(photons['dir'])), expected_dir)

    # This test could be expanded but the numbers in the 'expected' arrays have not been correctly calculated yet.
    #
    # # confirm reflection probability
    # polarizedFile = ascii.read('./marxs/optics/data/ALSpolarization2.txt')
    # correctTestPolar = np.interp(1.23984282 / 3.02, polarizedFile['Photon energy'] / 1000, polarizedFile['Polarization'])
    # expected_prob = np.array([0.0581 * np.exp(-0.5) * 1. / correctTestPolar, 0.0042 * 0., 0.])
    # assert np.allclose(np.array(photons['probability']), expected_prob)
    #
    # # confirm correct new polarization
    # expected_pol = np.array([[0., 1., 0., 0], [1., 1.5, 0., 0], [0., 0., 1., 0], [0., 0., 1., 0]])
    # #expected_pol[1,0:3] = np.cross(photons['dir'][1,0:3] / np.linalg.norm(photons['dir'][1,0:3]), expected_pol[0,0:3])
    # for i in range(0, 2):
    #   assert np.allclose(np.array(photons['polarization'][i]), expected_pol[i]) or np.allclose(np.array(photons['polarization'][i]), -expected_pol[i])

@pytest.mark.parametrize("angle,avgpol", [(0., 0.5), (np.pi/2, 0.)])
def test_double_reflection(angle, avgpol):
    '''Two mirrors

    The first mirror polarizes the light and the second
    one perfectly reflects that polarized light or perfectly absorbs it.
    '''
    pos = np.tile([1., 0., 0., 1.], (8, 1))
    dir = np.tile([-1., 0., 0., 0.], (8, 1))

    ang = np.arange(0, 2. * np.pi, np.pi / 4)
    polarization = np.vstack([np.zeros(8), np.sin(ang), np.cos(ang), np.zeros(8)]).T

    photons = Table({'pos': pos,
                     'dir': dir,
                     'energy': np.ones(8),
                     'polarization': polarization,
                     'probability': np.ones(8)})
    ml1 = FlatBrewsterMirror(orientation=euler.euler2mat(np.pi / 4, 0, 0, 'szxy'))
    ml2 = FlatBrewsterMirror(position=[0, 1, 0],
                             orientation=euler.euler2mat(-np.pi / 4, angle, 0, 'szyx'))
    photons = ml1(photons)
    photons = ml2(photons)
    assert np.isclose(np.mean(photons['probability']), avgpol)
