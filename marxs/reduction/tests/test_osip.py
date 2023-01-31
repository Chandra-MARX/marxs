# Licensed under GPL version 3 - see LICENSE.rst
from tempfile import TemporaryDirectory
from glob import glob
from os.path import join as pjoin
import numpy as np
import astropy.units as u
from astropy.table import Table, QTable

from marxs.missions.arcus import arfrmf
from marxs.missions.arcus.ccd import CCDRedist
from ..osip import FixedWidthOSIP, FixedFractionOSIP, FractionalDistanceOSIP

import pytest


ccd_redist = CCDRedist()
osipf = FixedWidthOSIP(40 * u.eV, ccd_redist=ccd_redist)
osipp = FixedFractionOSIP(0.7, ccd_redist=ccd_redist)
osipd = FractionalDistanceOSIP(ccd_redist=ccd_redist)


@pytest.mark.parametrize("thisosip", [osipf, osipp, osipd])
def test_osip_tab_returns(thisosip):
    '''Check return format and unit'''
    wave = [20, 21, 22, 23] * u.Angstrom
    tab = thisosip.osip_tab(wave, 2)

    assert tab.shape == (2, 4)
    assert tab.unit.physical_type == 'energy'
    assert np.all(tab.value > 0)


def test_osip_factor_width_unit():
    '''check units work for osip_width'''
    osip1 = FixedWidthOSIP(40 * u.eV, ccd_redist=ccd_redist)
    osip2 = FixedWidthOSIP(0.04 * u.keV, ccd_redist=ccd_redist)
    assert np.all(osip1.osip_factor([10] * u.Angstrom, -5, -5) ==
                  osip2.osip_factor([10] * u.Angstrom, -5, -5))


@pytest.mark.parametrize("thisosip", [osipf, osipp, osipd])
def test_osip_factor_wave_unit(thisosip):
    '''Check input works for wave or energy'''
    wave = [20, 21, 22, 23] * u.Angstrom
    energ = wave.to(u.keV, equivalencies=u.spectral())
    assert np.all(thisosip.osip_factor(wave, -5, -5) ==
                  thisosip.osip_factor(energ, -5, -5))


def test_osip_factor():
    '''test extreme values that do not depend on CCD resolution'''
    assert FixedWidthOSIP(0 * u.eV, ccd_redist=ccd_redist).osip_factor([10] * u.Angstrom, -5, -5) == 0
    wide = FixedWidthOSIP(10 * u.keV, ccd_redist=ccd_redist)
    assert np.allclose(wide.osip_factor([10] * u.Angstrom, -5, -5), 1)
    '''test with fixed sigma'''
    sig = QTable({'energy': [0, 1, 2] * u.keV, 'sigma': [40, 40, 40] * u.eV})
    myosip = FixedWidthOSIP(40 * u.eV,
                            ccd_redist=CCDRedist(sig))

    assert np.allclose(myosip.osip_factor([10] * u.Angstrom, -5, -5),
                       0.6827, rtol=1e-4)


@pytest.mark.parametrize("thisosip", [osipf, osipp, osipd])
def test_osip_factor_orders_on_different_sides(thisosip):
    '''If one order is positive and the other negative, then
    the signal is diffracted to opposite sides, so there is no
    contamination.
    '''
    assert thisosip.osip_factor([10] * u.Angstrom, -5, 5) == 0
    assert thisosip.osip_factor([10] * u.Angstrom, 1, -1) == 0
    assert thisosip.osip_factor([10] * u.Angstrom, 1, 0) == 0


@pytest.mark.parametrize("thisosip", [osipf, osipp, osipd])
def test_symmetry(thisosip):
    '''offset_orders +1 and -1 should have the same OSIP factors
    if OSIP is symmetric'''
    up = thisosip.osip_factor([10, 20, 30] * u.Angstrom, -5, -6)
    down = thisosip.osip_factor([10, 20, 30] * u.Angstrom, -5, -4)
    assert np.allclose(up, down, atol=1e-4)





@pytest.mark.parametrize("frac", [.5, .7, 1.])
def test_FractionalDistance_boundaries(frac):
    '''Test the boundaries of the extraction regions'''
    osipd = FractionalDistanceOSIP(frac, ccd_redist=ccd_redist)
    mlamgrid = [120, 130, 140] * u.Angstrom


    # Make comparison at the same m lambda position, i.e. the same poisition
    # on the chip
    tab5 = osipd.osip_tab(mlamgrid / 5, -5).to(u.eV, equivalencies=u.spectral())
    tab6 = osipd.osip_tab(mlamgrid / 6, -6).to(u.eV, equivalencies=u.spectral())

    # At the same mlambda, the dE does not depend on order
    assert u.allclose(tab5, tab6)

    # Now actually get the dE
    mlam5en = (mlamgrid / 5).to(u.eV, equivalencies=u.spectral())
    mlam6en = (mlamgrid / 6).to(u.eV, equivalencies=u.spectral())

    # Areas covered by extraction region
    a5 = tab5[0, :]
    a6 = tab6[1, :]

    assert np.allclose((a5 + a6) / (mlam6en - mlam5en), frac)


def test_Fixed_width_width():
    tab = osipf.osip_tab([10, 20, 30] * u.Angstrom, 5)
    assert u.allclose(tab, 40 * u.eV)


def test_Fixed_Fraction():
    '''Test FixedFractionOSIP end-to-end by actually making and
    changing a file'''

    with TemporaryDirectory() as tmpdirname:
        # -5 is last, because we need that for the tests below
        for o in [-4, -6, -5]:
            arf = arfrmf.mkarf([23, 24, 25, 26] * u.Angstrom, o)
            arf = arfrmf.tagversion(arf)

            basearf = pjoin(tmpdirname,
                            arfrmf.filename_from_meta('arf', **arf.meta))
            arf.write(basearf)

        osipp = FixedFractionOSIP(0.8, ccd_redist=ccd_redist,
                                  filename_from_meta=arfrmf.filename_from_meta)
        osipp.apply_osip_all(tmpdirname, tmpdirname, [-5],
                             filename_from_meta_kwargs={'ARCCHAN': '1111'})

        arfcenter = Table.read(glob(pjoin(tmpdirname, '*-5*-5*'))[0],
                               format='fits')
        assert np.isclose(arfcenter.meta['OSIPFAC'], 0.8)
        arfup = Table.read(glob(pjoin(tmpdirname, '*-5*-4*'))[0],
                           format='fits')
        assert np.all(arfup['OSIPFAC'] < .1)

        # Narrower OSIP region gives lower effective area
        osipp = FixedFractionOSIP(0.7, ccd_redist=ccd_redist,
                                  filename_from_meta=arfrmf.filename_from_meta)
        osipp.apply_osip_all(tmpdirname, tmpdirname, [-5], outroot='root',
                             filename_from_meta_kwargs={'ARCCHAN': '1111'})

        # To help debug the test:
        allfiles = glob(pjoin(tmpdirname, '*'))

        arfcenter7 = Table.read(glob(pjoin(tmpdirname, 'root*-5*-5*'))[0],
                                format='fits')
        assert np.isclose(arfcenter7.meta['OSIPFAC'], 0.7)
        assert u.allclose(arfcenter7['SPECRESP'],
                          0.7 / .8 * arfcenter['SPECRESP'])
        arfup7 = Table.read(glob(pjoin(tmpdirname, 'root*-5*-4*'))[0],
                            format='fits')
        assert np.all(arfup7['OSIPFAC'] < arfup['OSIPFAC'])
