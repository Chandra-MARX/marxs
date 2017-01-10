from io import StringIO
import numpy as np
from numpy.random import random
from astropy.table import Table
from transforms3d import axangles, affines
import pytest

from ..grating import (FlatGrating, CATGrating,
                       OrderSelector, EfficiencyFile)
from ...math.pluecker import h2e
from ... import energy2wave
from ...utils import generate_test_photons

def test_zeros_order():
    '''Photons diffracted into order 0 should just pass through'''
    photons = Table({'pos': random((10, 4)) * 10 - 5,
                     'dir': random((10, 4)) * 10 - 5,
                     'energy': random(10),
                     'polarization': random(10),
                     'probability': np.ones(10),
                     })
    # Make sure homogeneous coordiantes are valid
    # and infall is normal
    photons['pos'][:, 3] = 1
    photons['dir'][:, 1:] = 0

    p = photons.copy()
    g0 = FlatGrating(d=1./500.,
                     order_selector=OrderSelector([0]),
                     zoom=np.array([1., 5., 5.]))
    p = g0(p)
    # Direction unchanged
    d_in = h2e(photons['dir'])
    d_out = h2e(p['dir'])
    # normalize
    d_in =  d_in / np.sqrt(np.sum(d_in**2, axis=-1))[:, None]
    d_out =  d_out / np.sqrt(np.sum(d_out**2, axis=-1))[:, None]
    assert np.allclose(d_in, d_out)
    # all intersection points in y-z plane
    assert np.allclose(p['pos'][:, 0], 0.)
    # no offset between old and new ray
    assert np.allclose(np.cross(h2e(photons['pos']) - h2e(p['pos']), h2e(p['dir'])), 0)

def test_translation_invariance():
    '''For homogeneous gratings, the diffraction eqn does not care where the ray hits.'''
    photons = generate_test_photons(2)
    photons['dir'] = np.tile(np.array([1., 2., 3., 0.]), (2, 1))
    pos = np.tile(np.array([1., 0., 0., 1.]), (2, 1))
    delta_pos = np.array([0, .23, -.34, 0.])
    pos[1,:] += delta_pos
    photons['pos'] = pos
    for order in [-1, 0, 1]:
        g = FlatGrating(d=1./500, order_selector=OrderSelector([order]), zoom=20)
        p = g(photons.copy())
        assert np.all(p['order'] == order)
        assert np.allclose(p['pos'][:, 0], 0)
        assert np.allclose(p['dir'][0, :], p['dir'][1, :])
        assert np.allclose(p['pos'][0, :], p['pos'][1, :] - delta_pos)

def test_angle_dependence():
    '''Test the grating angle for non-perpendicular incidence.

    For this test, I write the grating equation in its simple traditional form:
    :math:`m \lambda = \delta s = d sin(\alpha)`.

    If the incoming ray is not perpendicular to the grating, we use the angle beta
    between the incoming ray and the grating normal. The grating equation than looks
    like this for m = 1:
    :math:`\delta s = \delta s_2 - \delta s_1 = d sin(\alpha) - d sin(\beta)`.
    '''
    d = 0.001
    beta = np.deg2rad([0., 5., 10., 20.])
    dir = np.zeros((len(beta), 4))
    dir[:, 1] = np.sin(beta)
    dir[:, 0] = -np.cos(beta)
    pos = np.ones((len(beta), 4))
    photons = Table({'pos': pos,
                     'dir': dir,
                     'energy': np.ones(len(beta)),
                     'polarization': np.ones(len(beta)),
                     'probability': np.ones(len(beta)),
                     })
    g = FlatGrating(d=0.001,  order_selector=OrderSelector([1]), zoom=20)
    p = g(photons)
    # compare sin alpha to expected value: sin(alpha) = sin(beta) + lambda/d
    alpha = np.arctan2(p['dir'][:, 1], -p['dir'][:, 0])
    assert np.allclose(np.sin(alpha), np.sin(beta) + 1.2398419292004202e-06/d)

def test_order_dependence():
    '''For small theta, the change in direction is m * dtheta'''
    photons = Table({'pos': np.ones((5,4)),
                     'dir': np.tile([1., 0, 0, 0], (5,1)),
                     'energy': np.ones(5),
                     'polarization': np.ones(5),
                     'probability': np.ones(5),
                     })
    def mock_order(x, y, z):
        return np.array([-2, -1, 0, 1, 2]), np.ones(5)
    g = FlatGrating(d=1./500, order_selector=mock_order)
    p = g(photons)
    # grooves run in z direction
    assert np.allclose(p['dir'][:, 2], 0.)
    # positive and negative orders are mirrored
    assert np.allclose(p['dir'][[0,1], 1], - p['dir'][[4,3], 1])
    # order 2 is twice as far as order 1 (for theta  small)
    assert np.abs(p['dir'][4, 1] / p['dir'][3, 1] - 2 ) < 0.00001

def test_energy_dependence():
    '''The grating angle should depend on the photon wavelength <-> energy.'''
    photons = Table({'pos': np.ones((5,4)),
                     'dir': np.tile([1., 0, 0, 0], (5,1)),
                     'energy': np.arange(1., 6),
                     'polarization': np.ones(5),
                     'probability': np.ones(5),
                     })
    g = FlatGrating(d=1./500, order_selector=OrderSelector([1]))
    p = g(photons)
    # grooves run in z direction
    assert np.allclose(p['dir'][:, 2], 0.)
    # n lambda = d sin(theta)
    lam = energy2wave / p['energy']
    theta = np.arctan2(p['dir'][:, 1], p['dir'][:, 0])
    assert np.allclose(1. * lam, 1./500. * np.sin(theta))

def test_blaze_dependence():
    '''Ensure that the blaze angle is passed to the order selector (regression test)'''
    def order_selector(energy, polarization, blaze):
        '''test function (unphysical)'''
        return np.floor(blaze * 10), np.ones_like(energy)

    photons = generate_test_photons(2)
    photons['dir'][1, :] = [-1,1, 0, 0]
    g = FlatGrating(d=1., order_selector=order_selector, zoom=5)
    p = g(photons)
    assert np.allclose(p['order'], [0, 7])

def test_groove_direction():
    '''Direction of grooves may not be parallel to z axis.'''
    photons = generate_test_photons(5)
    order1 = OrderSelector([1])

    g = FlatGrating(d=1./500, order_selector=order1)
    assert np.allclose(np.dot(g.geometry('e_groove'), g.geometry('e_perp_groove')), 0.)
    p = g(photons.copy())

    g1 = FlatGrating(d=1./500, order_selector=order1, groove_angle=.3)
    p1 = g1(photons.copy())

    pos3d = axangles.axangle2mat([1,0,0], .3)
    g2 = FlatGrating(d=1./500, order_selector=order1, orientation=pos3d)
    p2 = g2(photons.copy())

    pos4d = axangles.axangle2aff([1,0,0], .1)
    g3 = FlatGrating(d=1./500, order_selector=order1, groove_angle=.2, pos4d=pos4d)
    p3 = g3(photons.copy())


    def angle_in_yz(vec1, vec2):
        '''project in the y,z plane (the plane of the grating) and calculate angle.'''
        v1 = vec1[1:3]
        v2 = vec2[1:3]
        arccosalpha = np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))
        return np.arccos(arccosalpha)

    assert np.allclose(angle_in_yz(p1['dir'][0,:], p2['dir'][0, :]), 0)
    assert np.allclose(angle_in_yz(p3['dir'][0,:], p2['dir'][0, :]), 0)

    for px in [p1, p2, p3]:
            assert np.allclose(angle_in_yz(p['dir'][0,:], px['dir'][0, :]), 0.3)


def test_order_convention():
    dirs = np.zeros((3, 4))
    dirs[1, 2] = 0.01
    dirs[2, 2] = -0.01
    dirs[:, 0] = - 1
    photons = Table({'pos': np.ones((3, 4)),
                     'dir': dirs,
                     'energy': np.ones(3),
                     'polarization': np.ones(3),
                     'probability': np.ones(3),
                     })
    gp = FlatGrating(d=1./500, order_selector=OrderSelector([1]), zoom=2)
    p1 = gp(photons.copy())
    gm = FlatGrating(d=1./500, order_selector=OrderSelector([-1]), zoom=2)
    m1 = gm(photons.copy())
    assert np.all(p1['order'] == 1)
    assert np.all(m1['order'] == -1)
    # intersection point with grating cannot depend on order
    assert np.all(p1['pos'].data == m1['pos'].data)


def test_CAT_order_convention():
    dirs = np.array([[-1, 0., 0., 0],
                     [-1, 0.01, -0.01, 0],
                     [-1, 0.01, 0.01, 0],
                     [-1, -0.01, 0.01, 0],
                     [-1, -0.01, -0.01, 0]])
    photons = Table({'pos': np.ones((5, 4)),
                     'dir': dirs,
                     'energy': np.ones(5),
                     'polarization': np.ones(5),
                     'probability': np.ones(5),
                     })
    gp = CATGrating(d=1./5000, order_selector=OrderSelector([5]), zoom=2)
    p5 = gp(photons.copy())
    gm = CATGrating(d=1./5000, order_selector=OrderSelector([-5]), zoom=2)
    m5 = gm(photons.copy())
    for g in [gm, gp]:
        assert np.all(g.order_sign_convention(h2e(photons['dir'])) == np.array([1, -1, -1, 1, 1]))
    assert np.all(p5['dir'][1:3, 1] > 0)
    assert np.all(p5['dir'][3:, 1] < 0)
    assert np.all(m5['dir'][1:3, 1] < 0)
    assert np.all(m5['dir'][3:, 1] > 0)


def test_uniform_efficiency():
    '''Uniform efficiency should give results from -order to +order'''
    f = OrderSelector(np.arange(-2, 3))
    # Make many numbers, to reduce chance of random failure for this test
    testout = f(np.ones(10000), np.ones(10000))
    assert set(testout[0]) == set([-2, -1 , 0, 1, 2])
    assert len(testout[0]) == len(testout[1])


def test_OrderSelector_probabilities():
    '''Most tests here use a constant order. This test makes sure that the
    probability is properly used if passed in.
    '''
    f = OrderSelector([0,1], [.3, .6])
    # Make many numbers, to reduce chance of random failure for this test
    testout = f(np.ones(10000), np.ones(10000))
    assert set(testout[0]) == set([0, 1])
    assert np.allclose(testout[1], 0.9)
    assert (testout[0] == 0).sum() < 0.8 * (testout[0] == 1).sum()


def test_OrderSelector_failures():
    with pytest.raises(ValueError) as e:
        f = OrderSelector([1, 2, 3, 4], p=[0.1, 0.2, 0.3])
    assert 'does not match' in str(e.value)

    with pytest.raises(ValueError) as e:
        f = OrderSelector([1, 2, 3], p=[1, 0.2, 0.3])
    assert 'must be <= 1' in str(e.value)

    with pytest.raises(ValueError) as e:
        f = OrderSelector([1, 2, 3], p=[-0.1, 0.2, 0.3])
    assert 'negative' in str(e.value)


def test_constant_order():
    '''Constant order will give always the same order'''
    f = OrderSelector([2])
    testout = f(np.ones(15), np.ones(15))
    assert np.all(testout[0] == 2)
    assert len(testout[0]) == len(testout[1])


def test_EfficiencyFile():
    '''Read in the efficency from a file.

    To simplify testing, simulate an inout file using StringIO.
    '''
    data  = StringIO(u".5 .1 .1 .1 .4\n1. .1 .1 .1 .5\n1.5 0. .1 .0 .5")
    eff = EfficiencyFile(data, [1, 0, -1, -2])
    testout = eff(np.array([.7, 1.1]), np.zeros(2))
    assert np.allclose(testout[1], np.array([.7, .8]))

    testout = eff(1.6 * np.ones(1000), np.zeros(1000))
    assert np.allclose(testout[1], .6)
    assert set(testout[0]) == set([-2 , 0])
    assert (testout[0] == 0).sum()  < (testout[0] == -2).sum()


def test_CATGRating_misses():
    '''Regression test: CAT gratings that intersect only a fraction of rays
    returned an array of the wrong dimension from order_sign_convention.'''
    photons = generate_test_photons(5)
    photons['pos'][:, 1] = np.arange(5)

    cat = CATGrating(d=1./5000, order_selector=OrderSelector([5]), zoom=2)
    p = cat(photons)
    assert np.all(np.isnan(p['order'][3:]))
    assert np.all(np.isnan(p['grat_y'][3:]))


def test_grating_d_callable():
    '''d can be a function.'''
    photons = generate_test_photons(5)
    photons['pos'][0, 1] = 1.
    photons['pos'][1, 1] = -1.

    def dfunc(intercoos):
        darr = np.ones(intercoos.shape[0]) * 2e-4
        d = np.where(intercoos[:, 0]>=0, darr, darr / 2.)
        return d

    grat = FlatGrating(order_selector=OrderSelector([1]), zoom=2, d=dfunc)
    p = grat(photons)
    # negative one has half the grating constant and thus twice the angle
    assert np.abs(p['dir'][1, 1] / p['dir'][0, 1] - 2 ) < 0.00001

def test_change_position_after_init():
    '''Regression: In MARXS 0.1 the geometry could not be changed after init
    because a lot of stuff was pre-calculated in __init__. That should no
    longer be the case. This test is here to check that for gratings.'''
    photons = generate_test_photons(5)

    g1 = FlatGrating(d=1./500, order_selector=OrderSelector([1]), groove_angle=.3)
    p1 = g1(photons.copy())

    pos3d = axangles.axangle2mat([1,0,0], .3)
    trans = np.array([0., -.3, .5])
    # Make grating at some point
    g2 = FlatGrating(d=1./500, order_selector=OrderSelector([1]), position=trans)
    # then move it so that pos and groove direction match g1
    g2.pos4d = affines.compose(np.zeros(3), pos3d, 25.*np.ones(3))
    p2 = g2(photons.copy())

    assert np.allclose(p1['dir'], p2['dir'])
    assert np.allclose(p1['pos'], p2['pos'])

def test_gratings_are_independent():
    '''Regression test: Some grating properties are stored in a dict.
    This test ensures that the relevant numbers are in an instance attribute and not
    in a class attribute.
    However, we want to test only user visible properties, not hidden dicts like
    FlatGrating._geometry. So, compare grove dirs for to gratings.
    '''
    g1 = FlatGrating(d=1./500, order_selector=OrderSelector([1]), groove_angle=.3)
    g2 = FlatGrating(d=1./500, order_selector=OrderSelector([1]))
    assert not np.allclose(g1.geometry('e_groove'), g2.geometry('e_groove'))
