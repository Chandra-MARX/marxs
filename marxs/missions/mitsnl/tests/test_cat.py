# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
import astropy.units as u
from transforms3d.euler import euler2mat

from ..catgrating import *
from marxs.optics import CATGrating, OrderSelector, FlatDetector
from marxs.optics.scatter import RandomGaussianScatter
from marxs.utils import generate_test_photons

def test_nonparallelCATGrating_simplifies_to_CATGrating():
    '''With all randomness parameters set to 0, the results
    should match a CAT grating exactly.'''
    photons = generate_test_photons(50)
    # Mix the photons up a little
    scat = RandomGaussianScatter(scatter=0.01)
    photons = scat(photons)
    order_selector = OrderSelector([2])
    cat = CATGrating(order_selector=order_selector, d=0.001)
    npcat = NonParallelCATGrating(order_selector=order_selector, d=0.001)
    p1 = cat(photons.copy())
    p2 = npcat(photons)
    assert np.all(p1['dir'] == p2['dir'])


def test_scalingSitransparancy():
    '''The module has data for 1 mu Si and scales that to the depth of the
    grating. Test against known-good coefficient from CXRO.'''
    photons = generate_test_photons(1)
    cat = L1(order_selector=OrderSelector([-5]), relativearea=0.,
             depth=4*u.mu, d=0.00001)
    photons = cat(photons)
    assert np.isclose(photons['probability'][0], 0.22458, rtol=1e-4)


def test_L2_broadening():
    '''Compare calculated L2 broadening with the values calculated
    by Ralf Heilmann. Both calculation make simplications (e.g. assuming
    a circular hole), but should give the right order of magnitude for
    diffraction.'''
    photons = generate_test_photons(1000)
    photons['energy'] = 0.25  # 50 Ang
    l2 = L2Diffraction(l2_dims={'period': 0.966 * u.mm,
                                'barwidth': 0.1 * u.mm})
    photons = l2(photons)
    det = FlatDetector(position=[-1, 0, 0])
    photons = det(photons)
    sigma = np.std(photons['det_x'])
    # for small angles tan(alpha) = alpha
    # 1 arcsec = 4.8...e-6 rad
    # 0.5 is sigma calculated by Ralf Heilmann for a rectangular
    # aperture
    assert np.isclose(sigma, .5 * 4.8e-6, rtol=.5)


def test_L2_Abs():
    '''Check L2 absorption against numbers calculated by Ralf Heilmann.'''
    photons = generate_test_photons(1)
    l2 = L2Abs(l2_dims={'period': 0.966 * u.mm, 'bardepth': 0.5 * u.mm,
                        'barwidth': 0.1 * u.mm})
    photons = l2(photons)
    assert np.isclose(photons['probability'], 0.81, rtol=0.02)

def test_L2_Abs_angle():
    '''Test that that shadow area increases with angle.
    The comparison number was calculated by H. M. Guenther in response
    to Arcus DQ36.'''
    l2_dims = {'period': 0.966 * u.mm, 'bardepth': 0.5 * u.mm,
               'barwidth': 0.1 * u.mm}

    l2 = L2Abs(l2_dims=l2_dims)
    p1 = l2(generate_test_photons(1))

    l2 = L2Abs(l2_dims=l2_dims, orientation=euler2mat(np.deg2rad(1.8), 0, 0, 'szxy'))
    p2 = l2(generate_test_photons(1))

    assert np.isclose(p2['probability'][0] / p1['probability'][0], 0.979)
