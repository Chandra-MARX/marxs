# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
import astropy.units as u
from astropy.table import Table
import pytest

from ..import ogip

def generate_RMF():
    '''Make an RMF for testing. Using one possible method
    to generate all the required inputs.'''
    width = Table({'wave': [10, 13, 14, 15, 16] * u.Angstrom,
              'sigma_wave':  [0.1, 0.2, 0.2, .2, .2] * u.Angstrom})
    rmf = ogip.RMF.from_Gauss_sigma(np.arange(10, 15, 0.1) * u.Angstrom,
                                    width)
    return rmf


def test_get_row_exact_energy():
    '''Make an RMF and check a row'''
    sherpa = pytest.importorskip("sherpa")
    width = Table({'wave': [10, 13, 14, 15, 16] * u.Angstrom,
              'sigma_wave':  [0.1, 0.2, 0.2, .2, .2] * u.Angstrom})
    rmf = ogip.RMF.from_Gauss_sigma(np.arange(10, 15.5, 1) * u.Angstrom,
                                    width)
    energy = (10.5 * u.Angstrom).to(u.keV, equivalencies=u.spectral())
    row = rmf.row(energy)
    assert row['ENERG_LO'] * u.keV <= energy
    assert row['ENERG_HI'] * u.keV > energy
    assert row['N_GRP'] == 1
    assert np.min(row['MATRIX']) > 1e-6 # default threshold

    assert rmf.ebounds.meta['INSTRUME'] == 'unknown'


def test_threshold():
    sherpa = pytest.importorskip("sherpa")
    width = Table({'wave': [10, 13, 14, 15, 16] * u.Angstrom,
              'sigma_wave':  [0.1, 0.2, 0.2, .2, .2] * u.Angstrom})
    rmf = ogip.RMF.from_Gauss_sigma(np.arange(10, 15, 0.1) * u.Angstrom,
                                    width)
    energy = (11.5 * u.Angstrom).to(u.keV, equivalencies=u.spectral())
    row = rmf.row(energy)
    assert row["N_CHAN"] == [15]
    assert np.min(row['MATRIX']) > 1e-6 # default threshold
    assert np.sum(row['MATRIX']) == pytest.approx(1)

    # Cut at a higher threshold, so fewer elements included
    rmf2 = ogip.RMF.from_Gauss_sigma(np.arange(10, 15, 0.1) * u.Angstrom,
                                     width, threshold=1e-4)
    row2 = rmf2.row(energy)
    assert row2['N_CHAN'][0] < 19
    assert np.min(row2['MATRIX']) > 1e-4
    assert np.sum(row['MATRIX']) > np.sum(row2['MATRIX'])


def test_arr_to_rmf_matrix_row():
    sherpa = pytest.importorskip("sherpa")
    rmf = generate_RMF()

    arr = np.array([0, 1, 2, 0, 3.])
    ngrp, fchan, nchan, matrix = rmf.arr_to_rmf_matrix_row(arr, TLMIN_F_CHAN=0)
    assert ngrp == 2
    assert fchan == [1, 4]
    assert nchan == [2, 1]
    assert np.allclose(matrix, np.array([1, 2, 3]))

    ngrp, fchan, nchan, matrix = rmf.arr_to_rmf_matrix_row(arr, TLMIN_F_CHAN=1)
    assert ngrp == 2
    assert fchan == [2, 5]
    assert nchan == [2, 1]
    assert np.allclose(matrix, np.array([1, 2, 3]))

