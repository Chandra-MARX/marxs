'''Tests for calling marx the right way.

Test in this module are not meant to check that the marx c code works - that really
is the responsibility of the marx maintainer in a separate repository.
Instead, it does a few spot check to ensure that the c modules are called in the right
way. In particular, it holds regression tests for issues that were already discovered
and fixed.
'''
import numpy as np
from scipy.stats import ks_2samp

import marxs
import marxs.source
import marxs.source.source
import marxs.optics.marx

def test_noexplicettimedependence():
    '''In an older implementation, the first half of all photons went of
    mirror shell 0, which turned out to be due to ``sorted_index`` being a int,
    while marx expects an *unsigned* int.
    '''
    mysource = marxs.source.source.ConstantPointSource((30., 30.), flux=1., energy=1.)
    photons = mysource.generate_photons(1000)
    mypointing = marxs.source.source.FixedPointing(coords=(30, 30.))
    photons = mypointing.process_photons(photons)

    marxm = marxs.optics.marx.MarxMirror('./marxs/optics/hrma.par', position=np.array([0., 0,0]))
    photons = marxm.process_photons(photons)
    ks, p_value = ks_2samp(photons['mirror_shell'][:400], photons['mirror_shell'][600:])
    assert p_value > 1e-5
