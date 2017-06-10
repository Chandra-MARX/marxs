# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import pearsonr
import pytest

from ...source import FixedPointing, JitterPointing, PointSource


def test_reference_coordiante_system():
    '''Usually, rays that are on-axis will come in along the x-axis.
    Test a simulations with a different coordinate system.'''
    s = PointSource(coords=SkyCoord(12., 34., unit='deg'))
    photons = s.generate_photons(5)
    point_x = FixedPointing(coords=SkyCoord(12., 34., unit='deg'))
    p_x = point_x(photons.copy())
    assert np.allclose(p_x['dir'].data[:, 0], -1.)
    assert np.allclose(p_x['dir'].data[:, 1:], 0)

    xyz2zxy = np.array([[0., 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).T
    point_z = FixedPointing(coords=SkyCoord(12., 34., unit='deg'), reference_transform=xyz2zxy)
    p_z = point_z(photons.copy())
    assert np.allclose(p_z['dir'].data[:, 2], -1.)
    assert np.allclose(p_z['dir'].data[:, :2], 0)

def test_polarization_direction():
    '''Test that a FixedPointing correctly assigns linear polarization vectors.'''
    s = PointSource(coords=SkyCoord(187.4, 0., unit='deg'))
    photons = s.generate_photons(5)
    photons['polangle'] = (np.array([0., 90., 180., 270., 45.])) * u.deg.to(photons['polangle'].unit)
    point_x = FixedPointing(coords=SkyCoord(187.4, 0., unit='deg'))
    p_x = point_x(photons.copy())
    # For a simple polangle 0 = 180 and the direction does not matter.
    # However with a view to eliptical polarization in the future we want
    # to fix the absolute phase, too.
    assert np.allclose(p_x['polarization'].data[0, :], [0, 0, 1, 0])
    assert np.allclose(p_x['polarization'].data[1, :], [0, 1, 0, 0])
    assert np.allclose(p_x['polarization'].data[2, :], [0, 0, -1, 0])
    assert np.allclose(p_x['polarization'].data[3, :], [0, -1, 0, 0])
    assert np.allclose(p_x['polarization'].data[4, :], [0, 1/np.sqrt(2), 1/np.sqrt(2), 0])

    xyz2zxy = np.array([[0., 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).T
    point_z = FixedPointing(coords=SkyCoord(187.4, 0., unit='deg'), reference_transform=xyz2zxy)
    p_z = point_z(photons.copy())
    assert np.allclose(p_z['polarization'].data[0, :], [0, 1, 0, 0])
    assert np.allclose(p_z['polarization'].data[1, :], [1, 0, 0, 0])
    assert np.allclose(p_z['polarization'].data[4, :], [1/np.sqrt(2), 1/np.sqrt(2), 0, 0])

    # Photons pointing east with the same RA will have parallel polarization vectors
    # This is true for all pointing directions.
    s = PointSource(coords=SkyCoord(22.5, 0., unit='deg'))
    photons = s.generate_photons(5)
    photons['dec'] = [67., 23., 0., -45.454, -67.88]
    photons['polangle'] = (90. * u.deg).to(photons['polangle'].unit)
    point = FixedPointing(coords=SkyCoord(94.3, 23., unit='deg'))
    p = point(photons.copy())
    for i in range(1, len(p)):
        assert np.isclose(np.dot(p['polarization'][0], p['polarization'][i]), 1)

def test_jitter():
    '''test size and randomness of jitter'''
    n = 100000
    ra = np.random.rand(n) * 2 * np.pi
    # Note that my rays are not evenly distributed on the sky.
    # No need to be extra fancy here, it's probably even better for
    # a test to have more scrutiny around the poles.
    dec = (np.random.rand(n) * 2. - 1.) / 2. * np.pi
    time = np.arange(n)
    pol = np.ones_like(ra)
    prob = np.ones_like(ra)
    photons = Table([ra, dec, time, pol, prob],
                    names=['ra', 'dec', 'time', 'polangle', 'probability'])
    fixed = FixedPointing(coords = SkyCoord(25., -10., unit='deg'))
    jittered = JitterPointing(coords = SkyCoord(25., -10., unit='deg'),
                              jitter=1. * u.arcsec)
    p_fixed = fixed(photons.copy())
    p_jitter = jittered(photons)

    assert np.allclose(np.linalg.norm(p_fixed['dir'], axis=1), 1.)
    assert np.allclose(np.linalg.norm(p_jitter['dir'], axis=1), 1.)

    prod = np.sum(p_fixed['dir'] * p_jitter['dir'], axis=1)
    # sum can give values > 1 due to rounding errors
    # That would make arccos fail, so catch those here
    ind = prod > 1.
    if ind.sum() > 0:
        prod[ind] = 1.

    alpha = np.arccos(prod)
    # in this formula alpha will always be the abs(angle).
    # Flip some signs to recover input normal distribtution.
    alpha *= np.sign(np.random.rand(n)-0.5)
    # center?
    assert np.abs(np.mean(alpha)) * 3600. < 0.01
    # Is right size?
    assert np.std(np.rad2deg(alpha)) > (0.9 / 3600.)
    assert np.std(np.rad2deg(alpha)) < (1.1 / 3600.)
    # Does it affect y and z independently?
    coeff, p = pearsonr(p_fixed['dir'][:, 1] - p_jitter['dir'][:, 1],
                        p_fixed['dir'][:, 2] - p_jitter['dir'][:, 2])
    assert abs(coeff) < 0.01

@pytest.mark.parametrize('pointing', [FixedPointing, JitterPointing])
def test_polarization_perpendicular(pointing):
    '''Consistency: Polarization vector must always be perpendicular to dir.'''
    s = PointSource(coords=SkyCoord(0., 0., unit='deg'))
    photons = s.generate_photons(10)
    photons['ra'] = np.random.uniform(0, 360., len(photons))
    # Exclude +-90 deg, because not handling poles well
    photons['dec'] = np.random.uniform(-89.9, 89.9, len(photons))
    photons['polangle'] = np.random.uniform(0, 360., len(photons))
    point_x = FixedPointing(coords=SkyCoord(187.4, 0., unit='deg'))
    p_x = point_x(photons.copy())
    assert np.allclose(np.einsum('ij,ij->i', p_x['dir'].data,
                                 p_x['polarization'].data), 0)
