import numpy as np
from astropy.table import Table

from ..detector import FlatDetector
from ...test import closeornan

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
    photons = det.process_photons(photons)
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
