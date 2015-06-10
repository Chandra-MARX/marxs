from collections import OrderedDict

import numpy as np
import pytest

from ..aperture import BaseAperture, RectangleAperture
from ...source.source import ConstantPointSource, FixedPointing
from ..mirror import ThinLens
from ..detector import InfiniteFlatDetector
from ..grating import FlatGrating, uniform_efficiency_factory
from ..marx import MarxMirror

# Initialize all optical elements to be tested
all_oe = [ThinLens(focallength=100),
          RectangleAperture(),
          InfiniteFlatDetector(),
          FlatGrating(d=0.001, order_selector=uniform_efficiency_factory(0)),
          MarxMirror(parfile='marxs/optics/hrma.par'),
          ]

# Each elements will be used multiple times.
# Can I add a test to check that using them does not leave any
# extra attributes etc., but that they come out in a clean state?
# How would I do that?


# Make a test photon list
# Some of this should be separate tests, e.g. source position vs. pointing.
# Can I vary energy for e.g. grating?
mysource = ConstantPointSource((30., 30.), 1., 300.)
masterphotons = mysource.generate_photons(11)
mypointing = FixedPointing(coords=(30., 30.))
masterphotons = mypointing.process_photons(masterphotons)
myslit = RectangleAperture()
masterphotons = myslit.process_photons(masterphotons)


@pytest.fixture(autouse=True)
def photons():
    '''make a photons table - fresh for every test'''
    photons = masterphotons.copy()
    assert id(photons['dir']) != id(masterphotons['dir'])
    assert np.all(photons['dir'] == photons['dir'])
    return photons



mark = pytest.mark.parametrize


@mark('elem', all_oe)
class TestOpticalElementInterface:
    '''Test that all optical elements follow the interface convention.

    At the same time, these tests guarantee that the modules work at all
    (e.g. the can be initialized etc.). Not as good as checking for correctness
    (that has to happen in individual tests, but it's a start.
    '''
    def test_one_vs_many_in_call_signature(self, photons, elem):
        '''processing a single photon should give same result as photon list'''
        assert np.all(photons['dir'] == photons['dir'])
        p = photons[[0]]
        # Process individually
        single = photons[0]
        try:
            # For apertures input pos is ignored, but still needs to be there
            # to keep the function signature consistent.
            dir, pos, energy, pol, prob = elem.process_photon(single['dir'], single['pos'], single['energy'], single['polarization'])
        except NotImplementedError:
            # It's OK if only the a vectorized process_photons is implemented
            # but if both exist, they need to return the same answer.
            return
        # Process as table
        if isinstance(elem, BaseAperture):
            photons.remove_column('pos')
        p = elem.process_photons(p)

        assert np.all(dir == p['dir'][0])
        assert np.all(pos == p['pos'][0])
        assert energy == p['energy'][0]
        # pol can be nan for modules that cannot deal with polarization
        assert ((np.isnan(pol) and np.isnan(p['polarization'][0]))
                or (pol == p['polarization'][0]))
        assert prob == p['probability'][0]

    def test_desc(self, elem):
        '''every elements should return a Ordered Dict for description'''
        des = elem.describe()
        assert isinstance(des, OrderedDict)
        assert len(des) > 0
