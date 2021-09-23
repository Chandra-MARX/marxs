# Licensed under GPL version 3 - see LICENSE.rst

import numpy as np
import pytest

from ...utils import generate_test_photons
from ..base import _assign_col_value


def test_assign_col():
    'Test a scalar and a vector case'
    photons = generate_test_photons(5)
    # scaler index
    _assign_col_value(photons, 'energy', 2, 3.)
    assert np.allclose(photons['energy'], [1, 1, 3, 1, 1])
    # array index
    _assign_col_value(photons, 'energy', photons['energy'] < 2, 2)
    assert np.allclose(photons['energy'], [2, 2, 3, 2, 2])

    _assign_col_value(photons, 'probability', 1, .5)
    _assign_col_value(photons, 'probability', photons['energy'] < 2.5, .5)
    assert np.allclose(photons['probability'], [.5, .25, 1, .5, .5])


def test_assign_col_probability_wrong():
    'And exception should be thrown for probabilities below 0 or above 1'
    photons = generate_test_photons(5)

    with pytest.raises(ValueError):
        _assign_col_value(photons, 'probability', 1, 1.5)
    with pytest.raises(ValueError):
        _assign_col_value(photons, 'probability', photons['energy'] < 2.5, -.1)
