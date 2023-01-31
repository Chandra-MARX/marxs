# Licensed under GPL version 3 - see LICENSE.rst
from os.path import join as pjoin
from glob import glob
from tempfile import TemporaryDirectory

import numpy as np
import astropy.units as u
from astropy.table import Table

from marxs.reduction import osip
from marxs.missions.arcus.arcus import defaultconf
from marxs.missions.arcus import arfrmf
from marxs.missions.arcus.ccd import CCDRedist


def test_filename():
    n = arfrmf.filename_from_meta('arf', ARCCHAN='1111', ORDER=3)
    assert n == 'chan_all_+3.arf'

    n = arfrmf.filename_from_meta(ARCCHAN='1111', ORDER=3,
                           TRUEORD=-3, CCDORDER=5)
    assert n == 'chan_all_ccdord_+5_true_-3.fits'

    # After reading a fits file, keywords will be string.
    # Some may have been modified, so test mixture of int and string
    n = arfrmf.filename_from_meta(ARCCHAN='1111', ORDER='+3',
                           TRUEORD='-4', CCDORDER=-5)
    assert n == 'chan_all_ccdord_-5_true_-4.fits'

    # But and order is not confused by itself
    n = arfrmf.filename_from_meta(ARCCHAN='1111', ORDER=-3,
                           TRUEORD=-3, CCDORDER=-3)
    assert n == 'chan_all_ccdord_-3_true_-3.fits'


def test_onccd():
    '''Don't want to hardcode exact locations of CCDs in this test,
    because we might move around CCDs or aimpoints. So, instead, this test
    just checks a few generic properties and wavelengths that are wildly off
    chip.
    '''
    out = arfrmf.onccd(np.arange(200) * u.Angstrom, 1, '1')
    assert out.sum() > 0  # Some wavelength fall on chips
    assert out.sum() < len(out)  # but not all of them

    # chip gaps are different for different orders
    out1 = arfrmf.onccd(np.arange(25, 30, .001) * u.Angstrom, -5, '1')
    out2 = arfrmf.onccd(np.arange(25, 30, .001) * u.Angstrom, -6, '1')
    assert not np.all(out1 == out2)

    # chip gaps are at same position in m * lambda space
    out1 = arfrmf.onccd(np.arange(25, 30, .001) * u.Angstrom, -5, '1')
    out2 = arfrmf.onccd(np.arange(25, 30, .001) * u.Angstrom / 6 * 5, -6, '1')
    assert np.all(out1 == out2)

    # chip gaps are different for different opt ax
    out1 = arfrmf.onccd(np.arange(30, 35, .001) * u.Angstrom, -4, '1')
    out2 = arfrmf.onccd(np.arange(30, 35, .001) * u.Angstrom, -4, '2')
    assert not np.all(out1 == out2)


def test_arf_nonzero():
    '''Arcus design will evolve, so I don't want to test exact numbers here
    but if we are off my orders of magnitude that that's likely a code error.
    '''
    arf = arfrmf.mkarf([23, 24, 25, 26] * u.Angstrom, -5)
    assert np.all(arf['SPECRESP'] > 10 * u.cm**2)
    assert np.all(arf['SPECRESP'] < 400 * u.cm**2)


def test_arf_channels_addup():
    arfall = arfrmf.mkarf([33, 34, 35, 36] * u.Angstrom, -3)

    arflist = []
    for k in defaultconf['pos_opt_ax'].keys():
        arflist.append(arfrmf.mkarf([33, 34, 35, 36] * u.Angstrom, -3, channels=[k]))

    assert np.allclose(arfall['SPECRESP'],
                       sum([a['SPECRESP'] for a in arflist])
                       )


def test_mirrgrat():
    '''Test that mirr_grat works. Don't want to hardcode specific numbers
    (since they will change), but test a range here.
    '''
    aeff = arfrmf.mirr_grat([10, 20, 30] * u.Angstrom, -5)
    assert np.all(aeff < [10, 50, 100] * u.cm**2)
    assert np.all(aeff > [.1, 5., 10.] * u.cm**2)


def test_make_arf_osip():
    '''integration test: Generate and ARF and then apply an OSIP to it'''
    with TemporaryDirectory() as tmpdirname:
        # -5 is last, because we need that for the tests below
        for o in [-4, -6, -5]:
            arf = arfrmf.mkarf([23, 24, 25, 26] * u.Angstrom, o)
            arf = arfrmf.tagversion(arf)
            # check the default in mkarf
            assert arf.meta['ARCCHAN'] == '1111'

            basearf = pjoin(tmpdirname,
                            arfrmf.filename_from_meta('arf', **arf.meta))
            arf.write(basearf)

        osipp = osip.FixedFractionOSIP(0.7, ccd_redist=CCDRedist(),
                                       filename_from_meta=arfrmf.filename_from_meta)
        osipp.apply_osip_all(tmpdirname, tmpdirname, [-5],
                             filename_from_meta_kwargs={'ARCCHAN': '1111'})


        # For debugging: Keep list of all filenames in variable
        globres = glob(pjoin(tmpdirname, '*'))
        # there are three confused arfs now (to be read below)
        assert len(glob(pjoin(tmpdirname, '*-*-*'))) == 3

        # Check plots are created, but don't check content.
        # Not sure how to compare pdfs without too much work.
        assert len(glob(pjoin(tmpdirname, '*.pdf'))) > 0

        # using astropy.table here to be independent of the implementation of
        # ARF in arcus.reduction.ogip
        barf = Table.read(basearf, format='fits')

        # Using glob to get filename, to be independent of
        # filename_from_meta
        arfcenter = Table.read(glob(pjoin(tmpdirname, '*-5*-5*'))[0],
                               format='fits')
        assert np.isclose(arfcenter.meta['OSIPFAC'], 0.7)
        arfup = Table.read(glob(pjoin(tmpdirname, '*-5*-4*'))[0],
                           format='fits')
        arfdown = Table.read(glob(pjoin(tmpdirname, '*-5*-6*'))[0],
                             format='fits')
        # In this setup, some area falls between extraction regions
        assert np.all(barf['SPECRESP'] >
                      (arfcenter['SPECRESP'] +
                       arfup['SPECRESP'] +
                       arfdown['SPECRESP']))