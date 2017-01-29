# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from astropy.table import Table
import transforms3d

from ..detector import FlatDetector, CircularDetector
from ...tests import closeornan
from ...math.pluecker import h2e
from ...design import RowlandTorus

def test_pixelnumbers():
    pos = np.array([[0, 0., -0.25, 1.],
                    [0., 9.9, 1., 1.],
                    [0., 10.1, 1., 1.]])
    dir = np.ones((3,4), dtype=float)
    dir[:, 3] = 0.
    photons = Table({'pos': pos, 'dir': dir,
                     'energy': [1,2., 3.], 'polarization': [1.,2, 3.], 'probability': [1., 1., 1.]})
    det = FlatDetector(zoom=[1., 10., 5.], pixsize=0.5)
    assert det.npix == [40, 20]
    assert det.centerpix == [19.5, 9.5]
    photons = det(photons)
    assert closeornan(photons['det_x'], np.array([0, 9.9, np.nan]))
    assert closeornan(photons['det_y'], np.array([-0.25, 1., np.nan]))
    assert closeornan(photons['detpix_x'], np.array([19.5, 39.3, np.nan]))
    assert closeornan(photons['detpix_y'], np.array([9, 11.5, np.nan]))

    # Regression test: This was rounded down to [39, 39] at some point.
    det = FlatDetector(pixsize=0.05)
    assert det.npix == [40, 40]


def test_nonintegerwarning(recwarn):
    det = FlatDetector(zoom=np.array([1.,2.,3.]), pixsize=0.3)
    w = recwarn.pop()
    assert 'is not an integer multiple' in str(w.message)

def test_intersect_tube_miss():
    '''Rays passing at larger radius or through the center miss the tube.'''
    circ = CircularDetector()
    # Passing at larger radius
    intersect, interpos, inter_local = circ.intersect(np.array([[1., 0., .1, 0],
                                                                [-1., -1., 0., 0.]]),
                                                         np.array([[0, -1.1, 0., 1.],
                                                                   [2., 0., 0., 1.]]))
    assert np.all(intersect == False)
    assert np.all(np.isnan(interpos))
    assert np.all(np.isnan(inter_local))

    # Through the center almost parallel to z-axis
    intersect, interpos, inter_local = circ.intersect(np.array([[0., 0.1, 1., 0]]),
                                                         np.array([[0.5, 0.5, 0., 1.]]))
    assert np.all(intersect == False)
    assert np.all(np.isnan(interpos))
    assert np.all(np.isnan(inter_local))

    # Repeat with a tube that's moved to make sure we did not mess up local and
    # global coordinates in the implementation
    circ = CircularDetector(position=[0.8, 0.8, 1.2])
    intersect, interpos, inter_local = circ.intersect(np.array([[1., 0., .0, 0],
                                                                [1., 0., 0., 0.]]),
                                                         np.array([[0., 0., 0., 1.],
                                                                   [0., -0.3, 1., 1.]]))
    assert np.all(intersect == False)
    assert np.all(np.isnan(interpos))
    assert np.all(np.isnan(inter_local))

def test_intersect_tube_hitmiss_zoomed():
    '''Rays that hit a one tube, but miss a shorter one'''
    dir = np.array([[-.1, 0., 0., 0], [1., 0, -0.2, 0.]])
    pos = np.array([[20., 0., 0.8, 1.], [2., .0, 0.4, 1.]])

    circ = CircularDetector(zoom=[1, 1, 1])
    intersect, interpos, inter_local = circ.intersect(dir, pos)
    assert np.all(intersect == True)

    circ = CircularDetector(zoom=[1, 1, .5])
    intersect, interpos, inter_local = circ.intersect(dir, pos)
    assert np.all(intersect == False)
    assert np.all(np.isnan(interpos))
    assert np.all(np.isnan(inter_local))

def test_intersect_tube_2points():
    dir = np.array([[-.1, 0., 0., 0], [-0.5, 0, 0., 0.], [-1., 0., 0., 0.]])
    pos = np.array([[50., 0., 0., 1.], [10., .5, 0., 1.], [2, 0, .3, 1.]])

    circ = CircularDetector(phi_offset=-np.pi)
    intersect, interpos, inter_local = circ.intersect(dir, pos)
    assert np.all(intersect == True)
    assert np.allclose(h2e(interpos), np.array([[-1., 0., 0.],
                                                [-np.sqrt(0.75), 0.5, 0.],
                                                [-1, 0., .3]]))
    assert np.allclose(inter_local, np.array([[0., -np.arcsin(0.5), 0.],
                                              [0., 0., 0.3]]).T)

def test_intersect_tube_2points_zoomed():
    '''inter_local should be the same if we make the tube longer in z'''
    dir = np.array([[-.1, 0., 0., 0], [-0.5, 0, 0., 0.], [-1., 0., 0., 0.]])
    pos = np.array([[50., 0., 0., 1.], [10., .5, 0., 1.], [2, 0, .3, 1.]])

    circ = CircularDetector(phi_offset=-np.pi, zoom=[1, 1., 5.])
    intersect, interpos, inter_local = circ.intersect(dir, pos)
    assert np.all(intersect == True)
    assert np.allclose(h2e(interpos), np.array([[-1., 0., 0.],
                                                [-np.sqrt(0.75), 0.5, 0.],
                                                [-1, 0., .3]]))
    assert np.allclose(inter_local, np.array([[0., -np.arcsin(0.5), 0.],
                                              [0., 0., 0.3]]).T)

def test_intersect_tube_2points_outside():
    '''Now hit the outside. Looks very similar, if we remove the phi_offset, too.'''
    dir = np.array([[-.1, 0., 0., 0], [-0.5, 0, 0., 0.], [-1., 0., 0., 0.]])
    pos = np.array([[50., 0., 0., 1.], [10., .5, 0., 1.], [2, 0, .3, 1.]])

    circ = CircularDetector(inside=False)
    intersect, interpos, inter_local = circ.intersect(dir, pos)
    assert np.all(intersect == True)
    assert np.allclose(h2e(interpos), np.array([[1., 0., 0.],
                                                [np.sqrt(0.75), 0.5, 0.],
                                                [1, 0., .3]]))
    assert np.allclose(inter_local, np.array([[0., np.arcsin(0.5), 0.],
                                              [0., 0., 0.3]]).T)

def test_intersect_tube_2points_translation():
    '''Repeat with a tube that's moved and zoomed to make sure we
    did not mess up local and global coordinates in the implementation
    '''
    circ = CircularDetector(position=[0.8, 0.8, 1.2], zoom=[1, 1, 2])
    intersect, interpos, inter_local = circ.intersect(np.array([[1., 0., .0, 0],
                                                                [1., 0., 0., 0.]]),
                                                         np.array([[0., 0., 0., 1.],
                                                                   [0., 0., 1., 1.]]))
    assert np.all(intersect == True)
    assert np.allclose(h2e(interpos), np.array([[1.4, 0., 0.],
                                                [1.4, 0., 1.]]))

def test_tube_parametric():
    '''Generate points on surface using parametic. Then make rays for intersection
    which should return those points as intersection points.'''
    circ = CircularDetector(zoom=[2., 3., 4.])
    parametric = circ.parametric_surface(phi=[-.1, 0., 1.])
    parametric = parametric.reshape((6, 4))
    # select dir so that it's in the right direction to recover these points.
    dir = np.tile([1, 0, 0, 0], (6, 1))
    intersect, interpos, inter_local = circ.intersect(dir, parametric)
    assert np.allclose(parametric, interpos)

def test_CircularDetector_from_Rowland():
    '''If a circular detector is very narrow, then all points should be
    very close to the Rowland torus.'''
    rowland = RowlandTorus(R=6e4, r=5e4, position=[123., 345., -678.],
                           orientation=transforms3d.euler.euler2mat(1, 2, 3, 'syxz'))
    detcirc = CircularDetector.from_rowland(rowland, width=1e-6)
    phi = np.mgrid[0:2.*np.pi:360j]
    points = detcirc.parametric_surface(phi)
    # Quartic < 1e5 is very close for these large values of r and R.
    assert np.max(np.abs(rowland.quartic(h2e(points)))) < 1e5
