import numpy as np
from scipy.stats import kstest
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord

from transforms3d.axangles import axangle2aff

import pytest

import marxs
import marxs.optics
import marxs.optics.base
import marxs.optics.aperture
import marxs.source
from marxs.utils import generate_test_photons

@pytest.fixture(autouse=True)
def photons1000():
    '''Make a list of photons parallel to optical axis'''
    mysource = marxs.source.PointSource(coords=SkyCoord(30., 30., unit="deg"), energy=1., flux=300.)
    np.random.seed(0)
    p = mysource.generate_photons(1000)
    mypointing = marxs.source.FixedPointing(coords=(30., 30.))
    return mypointing.process_photons(p)


def test_pos4d_no_transformation():
    '''Test coordinate transforms on initialization of optical elements.

    This test checks that the appropriate transformations are done.
    It should be extended, once derived properties (e.g. a plane constructed
    from transformed points are implemented.
    '''
    oe = marxs.optics.aperture.RectangleAperture()
    assert np.all(oe.pos4d == np.eye(4))
    assert np.all(oe.geometry('center') == np.array([0, 0, 0, 1]))
    assert np.all(oe.geometry('e_y') == np.array([0, 1, 0, 0]))
    assert np.all(oe.geometry('e_z') == np.array([0, 0, 1, 0]))


def test_pos4d_rotation():
    rotation = axangle2aff(np.array([1, 1, 0]), np.deg2rad(90))
    oe = marxs.optics.aperture.RectangleAperture(orientation=rotation[:3,:3])
    assert np.all(oe.pos4d == rotation)
    assert np.all(oe.geometry('center') == np.array([0, 0, 0, 1]))
    assert np.allclose(oe.geometry('e_y'), np.array([.5, .5, 1. / np.sqrt(2), 0]))
    assert np.allclose(oe.geometry('e_z'), np.array([1., -1, 0, 0]) / np.sqrt(2))


def test_pos4d_translation():
    oe = marxs.optics.aperture.RectangleAperture(position = np.array([-2, 3, 4.3]))
    expectedpos4d = np.eye(4)
    expectedpos4d[:3, 3] = np.array([-2, 3, 4.3])
    assert np.all(oe.pos4d == expectedpos4d)
    assert np.all(oe.geometry('center') == np.array([-2, 3, 4.3, 1]))
    assert np.all(oe.geometry('e_y') == np.array([0, 1, 0, 0]))
    assert np.all(oe.geometry('e_z') == np.array([0, 0, 1, 0]))



mark = pytest.mark.parametrize

all_slits = [marxs.optics.aperture.RectangleAperture(zoom=2),
             marxs.optics.aperture.RectangleAperture(zoom=[2,2,2]),
            ]

@mark('myslit', all_slits)
def test_pos4d_transforms_slit(photons1000, myslit):
    '''Test coordinate transforms on initialization of optical elements.

    The initial 4D transforms should be done to any optical element.
    Here, I pick the entrance aperture for testing, because it places the
    positional vector of the plucker coordinates in a plane, independent
    of the initial values.
    '''

    p = myslit.process_photons(photons1000)
    assert np.allclose(p['pos'][:, 0], 0)
    assert kstest((p['pos'][:, 1] + 2) / 4, "uniform")[1] > 0.01
    assert kstest((p['pos'][:, 2] + 2) / 4, "uniform")[1] > 0.01


def test_pos4d_transforms_slit_rotated(photons1000):
    '''Test coordinate transforms on rotated entrance aperture.'''
    p = photons1000

    rotation = axangle2aff(np.array([0, 1, 0]), np.deg2rad(90))
    myslit = marxs.optics.aperture.RectangleAperture(orientation=rotation[:3, :3], zoom=0.5)
    p = myslit.process_photons(p)
    assert np.allclose(p['pos'][:, 2], 0)
    assert kstest(p['pos'][:, 0] + 0.5, "uniform")[1] > 0.01
    assert kstest(p['pos'][:, 1] + 0.5, "uniform")[1] > 0.01
    # Sometimes fails, because rays exactly parallel to aperture.
    # Find other example, e.g. 45 deg.
    # Check in aperture if that's the case and raise warning.


def test_photonlocalcoords_decorator():

    pos = np.array([[1, 0, 0, 1], [0, 1, 0, 1]])
    dir = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    photontab = Table([Column(data=pos, name='pos'),
                       Column(data=dir, name='dir')
                       ])

    class OEforTest(marxs.optics.base.OpticalElement):
        @marxs.optics.base.photonlocalcoords
        def functiontodecorate(self, photons):
            assert np.allclose(photons['dir'], dir)
            assert np.allclose(photons['pos'], np.array([[2., 2, 3, 1], [1, 3, 3, 1]]))
            return photons

    oetest = OEforTest(position=np.array([-1., -2, -3]))
    photontab = oetest.functiontodecorate(photontab)
    assert np.allclose(photontab['dir'], dir)
    assert np.allclose(photontab['pos'], pos)


def test_FlatStack():
    '''Run a stack of two elements and check that both are applied to the photons.'''
    fs = marxs.optics.FlatStack(position=[0, 5, 0], zoom=2,
                                elements=[marxs.optics.EnergyFilter, marxs.optics.FlatDetector],
                                keywords=[{'filterfunc': lambda x: 0.5},{}])
    # Check all layers are at the same position
    assert np.allclose(fs.geometry('center'), np.array([0, 5, 0, 1]))
    assert np.allclose(fs.elements[0].geometry('center'), np.array([0, 5, 0, 1]))
    assert np.allclose(fs.elements[1].geometry('center'), np.array([0, 5, 0, 1]))
    assert np.allclose(fs.pos4d, fs.elements[1].pos4d)

    p = generate_test_photons(5)
    # Photons 0, 1 miss the stack
    p['pos'][:, 1] = [0, 1, 3.1, 4, 5]
    fs.loc_coos_name = ['a', 'b']
    p = fs(p)
    assert np.allclose(p['probability'], [1, 1, .5, .5, .5])
    assert np.all(np.isnan(p['a'][:2]))
    assert np.allclose(p['a'][2:], [-1.9, -1., 0])
