# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from astropy.table import Table
import pytest
from ..baffles import Baffle, CircularBaffle


@pytest.mark.parametrize(
    "BaffleClass, photon_passed",
    [
        (Baffle, [True, False, True, True, True, True]),
        (CircularBaffle, [True, False, True, False, True, True]),
    ],
)
def test_photons_through(BaffleClass, photon_passed):
    """tests that photons go through or hit the baffle as predicted"""
    pos = np.array(
        [
            [1.0, 0.0, 0.0, 1],
            [1.0, 0.0, 0.0, 1],
            [1.0, 0.0, 0.0, 1],
            [1.0, 1.49, 0.29, 1],
            [1.0, 0.4, 0.0, 1],
            [1.0, 0, 0.2, 1],
        ]
    )
    dir = np.array(
        [
            [-1.0, 0.0, 0.0, 0],
            [-1.0, 1.4, 0.5, 0],
            [-1.0, -0.7, 0.2, 0],
            [-1.0, 0.0, 0.0, 0],
            [-1.0, 0.0, 0.0, 0],
            [-1.0, 0.0, 0.0, 0],
        ]
    )
    photons = Table(
        {
            "pos": pos,
            "dir": dir,
            "energy": [1, 1, 1, 1, 1, 1],
            "polarization": [1, 2, 3, 5, 6, 7],
            "probability": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    )
    baf = BaffleClass(zoom=np.array([1.0, 1.5, 0.3]))
    photons = baf(photons)
    expected = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) * np.array(
        photon_passed, dtype=int
    )
    assert photons["probability"] == pytest.approx(expected)
