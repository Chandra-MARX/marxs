import numpy as np
from scipy.stats import kstest
from astropy.table import Table, Column

from transforms3d.axangles import axangle2aff

import marxs
import marxs.optics
import marxs.optics.base
import marxs.optics.aperture
import marxs.source
import marxs.source.source


def test_pos4d_no_transformation():
    '''Test coordinate transforms on initialization of optical elements.

    This test checks that the appropriate transformations are done.
    It should be extended, once derived properties (e.g. a plane constructed
    from transformed points are implemented.
    '''
    oe = marxs.optics.aperture.SquareEntranceAperture(size=1)
    assert np.all(oe.pos4d == np.eye(4))
    assert np.all(oe.geometry['center'] == np.array([0, 0, 0, 1]))
    assert np.all(oe.geometry['e_y'] == np.array([0, 1, 0, 0]))
    assert np.all(oe.geometry['e_z'] == np.array([0, 0, 1, 0]))


def test_pos4d_rotation():
    rotation = axangle2aff(np.array([1, 1, 0]), np.deg2rad(90))
    oe = marxs.optics.aperture.SquareEntranceAperture(size=1, orientation=rotation[:3,:3])
    assert np.all(oe.pos4d == rotation)
    assert np.all(oe.geometry['center'] == np.array([0, 0, 0, 1]))
    assert np.allclose(oe.geometry['e_y'], np.array([.5, .5, 1. / np.sqrt(2), 0]))
    assert np.allclose(oe.geometry['e_z'], np.array([1., -1, 0, 0]) / np.sqrt(2))


def test_pos4d_translation():
    oe = marxs.optics.aperture.SquareEntranceAperture(size=1,position = np.array([-2, 3, 4.3]))
    expectedpos4d = np.eye(4)
    expectedpos4d[:3, 3] = np.array([-2, 3, 4.3])
    assert np.all(oe.pos4d == expectedpos4d)
    assert np.all(oe.geometry['center'] == np.array([-2, 3, 4.3, 1]))
    assert np.all(oe.geometry['e_y'] == np.array([0, 1, 0, 0]))
    assert np.all(oe.geometry['e_z'] == np.array([0, 0, 1, 0]))


def test_pos4d_transforms_slit():
    '''Test coordinate transforms on initialization of optical elements.

    The initial 4D transforms should be done to any optical element.
    Here, I pick the entrance aperture for testing, because it places the
    positional vector of the plucker coordinates in a plane, independent
    of the initial values.
    '''
    # Make this into a fixture if used by more than once.
    # Make a list of photons parallel to optical axis
    mysource = marxs.source.source.ConstantPointSource((30., 30.), 1., 300.)
    np.random.seed(0)
    p = mysource.generate_photons(1000)
    mypointing = marxs.source.source.FixedPointing(coords=(30., 30.))
    p = mypointing.process_photons(p)

    myslit = marxs.optics.aperture.SquareEntranceAperture(size=2)
    p = myslit.process_photons(p)
    assert np.allclose(p['pos'][:, 0], 0)
    assert kstest((p['pos'][:, 1] + 1) / 2, "uniform")[1] > 0.01
    assert kstest((p['pos'][:, 2] + 1) / 2, "uniform")[1] > 0.01


def test_pos4d_transforms_slit_rotated():
    '''Test coordinate transforms on rotated entrance aperture.'''
    # Make this into a fixture if used by more than once.
    # Make a list of photons parallel to optical axis
    np.random.seed(0)
    mysource = marxs.source.source.ConstantPointSource((30., 30.), 1., 300.)
    p = mysource.generate_photons(1000)
    mypointing = marxs.source.source.FixedPointing(coords=(30., 30.))
    p = mypointing.process_photons(p)

    rotation = axangle2aff(np.array([0, 1, 0]), np.deg2rad(90))
    myslit = marxs.optics.aperture.SquareEntranceAperture(size=1, orientation=rotation[:3, :3])
    p = myslit.process_photons(p)
    assert np.allclose(p['pos'][:, 2], 0)
    assert kstest(p['pos'][:, 0] + 0.5, "uniform")[1] > 0.01
    assert kstest(p['pos'][:, 1] + 0.5, "uniform")[1] > 0.01
    # Sometimes fails, because rays exactly parallel to aperture.
    # Find other example, e.g. 45 deg.
    # Check in aperture if that's the case and raise warning.


def test_photonlocalcoords_decorartor():

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
