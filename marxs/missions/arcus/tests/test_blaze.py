# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
import astropy.units as u
from .. import Arcus
from ..defaults import DefaultSource, DefaultPointing
from marxs.missions.arcus.utils import config as arcusconf

import pytest

@pytest.mark.skipif('data' not in arcusconf,
                    reason='Test requires Arcus CALDB')
def test_blaze():
    '''Check that photons have a reasonable blaze. We do not want to hardcode a
    particular value here but if they are all zero or they are all over the
    place then something is probably wrong.
    '''
    s = DefaultSource()
    point = DefaultPointing()
    p = s.generate_photons(10000 * u.s)
    p = point(p)
    arc = Arcus()
    p = arc(p)
    ind = np.isfinite(p['blaze']) & (p['probability'] > 0)

    assert np.all(np.rad2deg(p['blaze'][ind]) > 1.)
    assert np.all(np.rad2deg(p['blaze'][ind]) < 3.)
    assert np.all(np.std(np.rad2deg(p['blaze'][ind])) < .2)
