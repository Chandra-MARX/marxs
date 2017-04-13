import os
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from .... import optics, source
from ..hrma_py import Aperture, HRMA
from .. import HETG

parfile = os.path.join(os.path.dirname(optics.__file__), 'hrma.par')
marx = optics.MarxMirror(parfile)
aperpy = Aperture()
mirrpy = HRMA()


def test_area():
    '''Compare aperture area to MARX C code'''
    assert np.isclose(marx.area, aperpy.area, rtol=0.01, atol=1e-8 * u.mm**2)


def test_FWHM():
    '''Compare properties of PSF to MARX C code'''
    coords = SkyCoord(25., 45., unit=u.deg)
    src = source.PointSource(coords=coords)
    pointing = source.FixedPointing(coords=coords)
    p = src(1e4)
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
    assert np.isclose(ppy['probability'].sum(), pmarx['probability'].sum(), rtol=0.1)


def test_most_photons_hit_grating():
    '''If the radii of the mirror shells are correct, most photons should hit a
    facet in the HETG.'''
    coords = SkyCoord(25., 45., unit=u.deg)
    src = source.PointSource(coords=coords)
    pointing = source.FixedPointing(coords=coords)
    p = src(1e4)
    p = pointing(p)
    p = aperpy(p.copy())
    p = mirrpy(p)
    hetg = HETG()
    p = hetg(p)
    assert p['probability'][p['facet'] > 0].sum() > 5000.
