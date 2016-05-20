import numpy as np
import pytest
from ... import source, optics


@pytest.mark.parametrize("ra", [(0.,), (30.,), (-60.,)])
def test_PerfectLens(ra):
    mysource = source.PointSource((0., ra))
    mypointing = source.FixedPointing(coords=(0., 0.))
    myslit = optics.RectangleAperture(zoom=2)
    f = 1000
    lens = optics.PerfectLens(focallength=f,zoom=40)

    photons = mysource.generate_photons(11)
    photons = mypointing.process_photons(photons)
    photons = myslit.process_photons(photons)

    photons = lens.process_photons(photons)

    # How far away do I need to put a detector to hit the focal point?
    d = f * np.cos(np.deg2rad(ra))
    mdet = optics.FlatDetector(pixsize=0.01, position=np.array([-d, 0, 0]), zoom=1e5)
    photons = mdet.process_photons(photons)
    assert np.std(photons['det_x']) < 1e-4
    assert np.std(photons['det_y']) < 1e-4

    # Check that they actually diverge for other detector placement
    mdet = optics.FlatDetector(pixsize=0.01, position=np.array([-2 * d, 0, 0]), zoom=1e5)
    photons = mdet.process_photons(photons)
    assert np.std(photons['det_x']) > 1e-4
    assert np.std(photons['det_y']) > 1e-4
