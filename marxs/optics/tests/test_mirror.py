import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from ... import source, optics
from ...math.pluecker import h2e


@pytest.mark.parametrize("ra", [0., 30., -60.])
def test_PerfectLens(ra):
    mysource = source.PointSource(coords=SkyCoord(0., ra, unit="deg"))
    mypointing = source.FixedPointing(coords=SkyCoord(0., 0., unit='deg'))
    myslit = optics.RectangleAperture(zoom=2, position=[+100, 0, 0])
    f = 1000
    lens = optics.PerfectLens(focallength=f, zoom=400)

    photons = mysource.generate_photons(11)
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
