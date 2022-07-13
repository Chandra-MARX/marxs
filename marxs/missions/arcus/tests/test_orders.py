import numpy as np

from marxs.source import PointSource, FixedPointing
import astropy.units as u
from astropy.coordinates import SkyCoord
from marxs.math.utils import xyz2zxy
from .. import Arcus

import pytest

e = 0.5 * u.keV

mysource = PointSource(coords=SkyCoord(30. * u.deg, 30. * u.deg),
                       energy=e)
mypointing = FixedPointing(coords=SkyCoord(30 * u.deg, 30. * u.deg),
                           reference_transform=xyz2zxy)


@pytest.mark.parametrize("instrument", [Arcus(channels=['1']),
                                        Arcus(channels=['2m'])])
def test_orders_are_focussed(instrument):
    '''Check that the orders look reasonable.

    This test tries to be generic so that coordinate system etc. can be
    changed later, but still check that all light is focused to
    one point to detect error in setting up the mirrors.
    '''
    photons = mysource.generate_photons(2e4 * u.s)
    photons = mypointing(photons)
    photons = instrument(photons)

    for i in range(-12, 1):
        ind = (photons['order'] == i) & np.isfinite(photons['det_x']) & (photons['probability'] > 0)
        if ind.sum() > 100:
            assert np.std(photons['det_y'][ind & (photons['order_L1'] == 0)]) < 1
            assert np.std(photons['det_x'][ind]) < 1
            assert np.std(photons['det_x'][ind]) < np.std(photons['det_y'][ind])
            # But make sure that are not focussed too much
            # That would indicate that the scattering did not work
            assert np.std(photons['det_x'][ind]) > .04
            assert np.std(photons['det_y'][ind]) > .1


def test_zeroth_order_and_some_dispersed_orders_are_seen():
    '''test that both the zeroth order and some of the dispersed
    orders are positioned in the detector.
    '''
    photons = mysource.generate_photons(2e4 * u.s)
    photons = mypointing(photons)
    photons = Arcus()(photons)

    n_det = [((photons['order'] == i) & np.isfinite(photons['det_x'])).sum() for i in range(-12, 1)]
    assert n_det[-1] > 0
    assert sum(n_det[:9]) > 0


def test_two_optical_axes():
    '''Check that there are two position for the zeroth order.'''
    photons = mysource.generate_photons(1e5 * u.s)
    photons = mypointing(photons)
    photons = Arcus()(photons)
    i0 = (photons['order'] == 0) & np.isfinite(photons['det_x']) & (photons['probability'] > 0)

    assert i0.sum() > 250
    assert np.std(photons['det_y'][i0]) > 1
    assert np.std(photons['det_x'][i0]) > 1


@pytest.mark.parametrize("instrum, expected_area",
                         [(Arcus(), 400 * u.cm**2),
                          (Arcus(channels=['1']), 100 * u.cm**2),
                          (Arcus(channels=['2']), 100 * u.cm**2),
                          (Arcus(channels=['1m']), 100 * u.cm**2),
                          (Arcus(channels=['2m']), 100 * u.cm**2)])
def test_effective_area(instrum, expected_area):
    '''Surely, the effective area of Arcus will eveolve a little when the
    code is changed to accomendate e.g. a slightly different mounting
    for the gratings, but if the effective area drops or increases
    dramatically, that is more likely a sign for a bug in the code.
    '''
    photons = mysource.generate_photons(2e4 * u.s)
    photons = mypointing(photons)
    photons = instrum(photons)
    ind = np.isfinite(photons['det_x'])
    a_eff = np.sum(photons['probability'][ind]) / len(photons) * instrum.elements[0].area
    assert a_eff > 0.7 * expected_area
    assert a_eff < 1.5 * expected_area
