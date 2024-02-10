# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.interpolate import RectBivariateSpline

from ... import source, optics
from ...math.utils import h2e
from marxs.utils import generate_test_photons


@pytest.mark.parametrize("ra", [0., 30., -60.])
def test_PerfectLens(ra):
    mysource = source.PointSource(coords=SkyCoord(0., ra, unit="deg"))
    mypointing = source.FixedPointing(coords=SkyCoord(0., 0., unit='deg'))
    myslit = optics.RectangleAperture(zoom=2, position=[+100, 0, 0])
    f = 1000
    lens = optics.PerfectLens(focallength=f, zoom=400)

    photons = mysource.generate_photons(11 * u.s)
    photons = mypointing(photons)
    photons = myslit(photons)
    assert np.allclose(h2e(photons['pos'].data)[:, 0], 100.)

    photons = lens(photons)
    assert np.allclose(h2e(photons['pos'].data)[:, 0], 0.)

    # How far away do I need to put a detector to hit the focal point?
    d = f * np.cos(np.deg2rad(ra))
    mdet = optics.FlatDetector(pixsize=0.01, position=np.array([-d, 0, 0]), zoom=1e5)
    photons = mdet(photons)
    assert np.std(photons['det_x']) < 1e-4
    assert np.std(photons['det_y']) < 1e-4

    # Check that they actually diverge for other detector placement
    mdet = optics.FlatDetector(pixsize=0.01, position=np.array([-2 * d, 0, 0]), zoom=1e5)
    photons = mdet(photons)
    assert np.std(photons['det_x']) > 1e-4
    assert np.std(photons['det_y']) > 1e-4


def test_PerfectLens_offset():
    """Test a lens that is not centered on the optical axis."""
    lens = optics.PerfectLens(focallength=100, position=[0, 0, 10], zoom=400)
    lens2 = optics.PerfectLens(
        focallength=100, position=[0, 0, 0], zoom=400, d_center_optical_axis=-10
    )
    lens3 = optics.PerfectLens(
        focallength=100, position=[0, 0, 0], zoom=400, d_center_optical_axis=0
    )
    photons = generate_test_photons(1)
    p2 = generate_test_photons(1)
    p3 = generate_test_photons(1)
    photons = lens(photons)
    p2 = lens2(p2)
    p3 = lens3(p3)
    np.testing.assert_allclose(photons["dir"], p2["dir"])
    not np.allclose(photons["dir"], p3["dir"])


def test_PerfectLens_refl():
    """Test a reflection.

    This test sis a bit silly in the sense that it
    uses an unnecessarily complicated reflectivity function
    but then reflect at 0 deg."""
    xarr = np.linspace(-3, 3, 100)
    yarr = np.linspace(-3, 3, 100)
    xgrid, ygrid = np.meshgrid(xarr, yarr, indexing="ij")
    zdata = np.exp(-np.sqrt((xgrid / 2) ** 2 + ygrid**2))
    interp = RectBivariateSpline(xarr, yarr, zdata, kx=1, ky=1)

    lens = optics.PerfectLens(focallength=123.0, reflectivity_interpolator=interp)
    photons = generate_test_photons(1)
    photons = lens(photons)

    assert np.allclose(photons["energy"], 1)
    assert np.allclose(photons["polarization"], [0, 1, 0, 0])
    assert np.allclose(photons["probability"], 0.367205)
