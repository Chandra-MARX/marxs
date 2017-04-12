# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import pytest

from ..aperture import CircleAperture, RectangleAperture, MultiAperture
from ...utils import generate_test_photons
from ...source import FixedPointing
from ...base import GeometryError
from ...math.utils import h2e

def test_circle_phi():
    p = generate_test_photons(500)
    p.remove_column('pos')
    c = CircleAperture(phi=[0, np.pi / 2])
    p = c(p)
    assert np.all(p['pos'][1] >= 0)
    assert np.all(p['pos'][2] >= 0)

def test_circle_invalid_parameters():
    with pytest.raises(ValueError) as e:
        c = CircleAperture(phi=[0, 360.])
    assert 'Did you use degrees' in str(e.value)

    with pytest.raises(ValueError) as e:
        c = CircleAperture(phi=[0.2, -0.2])
    assert 'must be greater than' in str(e.value)

    with pytest.raises(ValueError) as e:
        c = CircleAperture(phi=[-.5, 2 * np.pi])
    assert 'less than 2 pi' in str(e.value)

    with pytest.raises(ValueError) as e:
        c = CircleAperture(r_inner=0.5, zoom=.2)
    assert 'must be less than' in str(e.value)

    with pytest.raises(GeometryError) as e:
        c = CircleAperture(zoom=[2, 2, 3])
    assert 'does not have the same size' in str(e.value)

def test_circle_radius():
    '''Check that all rays actually fall into the aperture area.'''
    p = generate_test_photons(500)
    p.remove_column('pos')
    c = CircleAperture(phi=[0, np.pi / 2], r_inner=15, zoom=25)
    p = c(p)
    assert np.all(p['pos'][1] >= 0)
    assert np.all(p['pos'][2] >= 0)
    rad = np.linalg.norm(h2e(p['pos']), axis=1)
    assert np.all(rad >= 15.)
    assert np.all(rad <= 25.)

def test_circle_area():
    '''Check that the area of rings and wedges comes out correctly.'''
    c = CircleAperture(zoom=[1, 2, 2])
    assert np.isclose(c.area, np.pi * 4 * u.mm**2, atol=1e-8*u.mm**2)

    c = CircleAperture(zoom=[1, 2, 2], r_inner=1)
    assert np.isclose(c.area, np.pi * 3 * u.mm**2, atol=1e-8*u.mm**2)

    c = CircleAperture(zoom=[1, 2, 2], phi = [1.3 * np.pi, 1.8 * np.pi])
    assert np.isclose(c.area, np.pi * u.mm**2, atol=1e-8*u.mm**2)

    c = CircleAperture(zoom=[1, 2, 2], phi = [1.3 * np.pi, 1.8 * np.pi], r_inner=1)
    assert np.isclose(c.area, np.pi * 0.75 * u.mm**2, atol=1e-8*u.mm**2)


def test_MultiAperture():
    p = generate_test_photons(1000)
    p.remove_column('pos')

    a1 = CircleAperture()
    a2 = RectangleAperture(position=[0, 4, 0], zoom=2)
    a = MultiAperture(elements=[a1, a2], id_col='aper')

    p = a(p)

    # All photons go through some aperture
    assert (p['aper'] == 0).sum() + (p['aper'] == 1).sum() == 1000
    # Most photons go through the bigger aperture
    assert (p['pos'][1:] > 1).sum() > 650

def test_geomarea_projection():
    '''When a ray sees the aperture under an angle the projected aperture size
    is smaller. This is accounted for by reducing the probability of this photon.'''
    photons = Table()
    photons['ra'] = [0., 45., 90., 135., 315.]
    photons['dec'] = np.zeros(5)
    photons['origin_coord'] = SkyCoord(photons['ra'], photons['dec'], unit='deg')
    photons['probability'] = np.ones(5)
    photons['time'] = np.arange(5)
    photons['polangle'] = np.zeros(5)

    fp = FixedPointing(coords=SkyCoord(0., 0., unit='deg'))
    photons = fp(photons)
    aper = RectangleAperture()
    p = aper(photons.copy())

    assert np.allclose(p['probability'], [1., 1./np.sqrt(2), 0, 0, 1./np.sqrt(2)])

    orientation = np.array([[0, 0, 1],[1, 0, 0],[0, 1, 0]])
    aper = RectangleAperture(orientation=orientation)
    p = aper(photons.copy())

    assert np.allclose(p['probability'], [ 0, 1./np.sqrt(2), 1., 1./np.sqrt(2), 0.])
