import os
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import pytest

from .... import optics, source
from ..hrma_py import Aperture, HRMA
from .. import HETG
from ....optics.marx import HAS_MARX

parfile = os.path.join(os.path.dirname(optics.__file__), "hrma.par")
aperpy = Aperture()
mirrpy = HRMA()


@pytest.mark.skipif("not HAS_MARX", reason="MARX C code not available")
def test_FWHM():
    '''Compare properties of PSF to MARX C code'''
    marx = optics.MarxMirror(parfile)

    coords = SkyCoord(25., 45., unit=u.deg)
    src = source.PointSource(coords=coords)
    pointing = source.FixedPointing(coords=coords)
    p = src(10 * u.ks)
    p = pointing(p)
    pmarx = marx(p.copy())
    ppy = aperpy(p.copy())
    ppy = mirrpy(ppy)

    fp = optics.FlatDetector()
    pmarx = fp(pmarx)
    ppy = fp(ppy)

    assert np.isclose(np.nanstd(pmarx['det_x']), np.nanstd(ppy['det_x']), rtol=0.25)
    assert np.isclose(np.nanstd(pmarx['det_y']), np.nanstd(ppy['det_y']), rtol=0.25)
    assert np.isclose(np.nanmean(ppy['det_x']), 0, atol=1e-3)


def test_most_photons_hit_grating():
    '''If the radii of the mirror shells are correct, most photons should hit a
    facet in the HETG.'''
    coords = SkyCoord(25., 45., unit=u.deg)
    src = source.PointSource(coords=coords)
    pointing = source.FixedPointing(coords=coords)
    p = src(10. * u.ks)
    p = pointing(p)
    p = aperpy(p.copy())
    p = mirrpy(p)
    hetg = HETG()
    p = hetg(p)
    assert p['probability'][p['facet'] > 0].sum() > 5000.
