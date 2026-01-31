# Licensed under GPL version 3 - see LICENSE.rst
'''Tests for calling marx the right way.

Test in this module are not meant to check that the marx c code works - that really
is the responsibility of the marx maintainer in a separate repository.
Instead, it does a few spot check to ensure that the c modules are called in the right
way. In particular, it holds regression tests for issues that were already discovered
and fixed.
'''
import numpy as np
from scipy.stats import ks_2samp
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
import pytest

import marxs
import marxs.source
import marxs.optics.marx


@pytest.mark.skipif(
    "not marxs.optics.marx.HAS_MARX", reason="MARX C code not available"
)
def test_noexplicettimedependence():
    '''In an older implementation, the first half of all photons went of
    mirror shell 0, which turned out to be due to ``sorted_index`` being a int,
    while marx expects an *unsigned* int.
    '''
    mysource = marxs.source.PointSource(coords=SkyCoord(30., 30., unit="deg"))
    photons = mysource.generate_photons(1 * u.ks)
    mypointing = marxs.source.FixedPointing(coords=SkyCoord(30, 30., unit='deg'))
    photons = mypointing(photons)

    marxm = marxs.optics.marx.MarxMirror(parfile=get_pkg_data_filename('hrma.par',
                                                   package='marxs.optics'),
                                         position=np.array([0., 0,0]))
    photons = marxm(photons)
    ks = ks_2samp(photons["mirror_shell"][:400], photons["mirror_shell"][600:])
    assert ks.pvalue > 1e-5
