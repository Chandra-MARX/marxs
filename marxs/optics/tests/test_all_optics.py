'''This module contains tests that are done for all opticial elements as opposed to tests that
apply to one class of elements only (e.g. gratings).
Tests here include simple things that guarantee that all elements comply with the MARXS interface
(e.g. each element must return a name and description) and more complex consistency checks
that are required for every type of optical element (e.g. the polarization vector must be
perpendicular to the photon propagation direction in each an every case).
'''
from collections import OrderedDict
import copy

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
import transforms3d.axangles

from .. import (RectangleAperture, ThinLens, FlatDetector, CircularDetector,
                FlatGrating, OrderSelector,
                MarxMirror, CircleAperture, MultiAperture)

from ..aperture import BaseAperture
from ...source import PointSource, FixedPointing
from ..base import _parse_position_keywords, FlatStack
from ...design import (RowlandTorus, GratingArrayStructure,
                       LinearCCDArray, RowlandCircleArray,
                       RectangularGrid)
from ..baffle import Baffle
from ..multiLayerMirror import MultiLayerMirror
from ...simulator import Sequence
from ...missions.chandra.hess import HETG
from ..scatter import RadialMirrorScatter
from ..filter import EnergyFilter


# Initialize all optical elements to be tested
mytorus = RowlandTorus(0.5, 0.5)

all_oe = [ThinLens(focallength=100),
          RectangleAperture(),
          CircleAperture(),
          MultiAperture(elements=[RectangleAperture(),
                                  CircleAperture(position=[0, 200, 0])]),
          FlatDetector(pixsize=2., zoom=100.),
          CircularDetector(),
          FlatGrating(d=0.001, order_selector=OrderSelector([0])),
          MarxMirror(parfile='marxs/optics/hrma.par'),
          GratingArrayStructure(mytorus, d_element=0.1, x_range=[0.5, 1.], radius=[0,.5],
                                elem_class=FlatGrating,
                                elem_args={'zoom':0.05, 'd':0.002,
                                           'order_selector': OrderSelector([1])
                                           }),
          LinearCCDArray(mytorus, d_element=0.05, x_range=[0., 0.5],
                          radius=[0., 0.5], phi=0., elem_class=FlatGrating,
                         elem_args={'zoom': 0.05, 'd':0.002,
                                    'order_selector': OrderSelector([2])
                                }),
          RowlandCircleArray(rowland=mytorus, elem_class=FlatGrating,
                             elem_args={'zoom': 0.04, 'd': 1e-3,
                                        'order_selector': OrderSelector([2])},
                             d_element=0.1,theta=[np.pi - 0.2, np.pi + 0.1]),
          RectangularGrid(rowland=mytorus, elem_class=FlatGrating,
                          elem_args={'zoom': 0.04, 'd': 1e-3,
                                     'order_selector': OrderSelector([-1, 0])},
                          d_element=0.1, x_range=[0.8, 1.1], y_range=[0, 0.1],
                          z_range=[0, 0.1]),

          Baffle(),
          # MLMirror assumes 45 deg angle and will return nan otherwise
          MultiLayerMirror('./marxs/optics/data/testFile_mirror.txt',
                           './marxs/optics/data/ALSpolarization2.txt',
                           orientation=transforms3d.axangles.axangle2mat([0,1,0], np.pi/4)),
          Sequence(elements=[]),
          HETG(),
          RadialMirrorScatter(inplanescatter=0.1),
          # not a useful filterfunc, but OK for testing with a few other dependencies
          EnergyFilter(filterfunc=lambda x: np.abs(np.cos(x))),
          FlatStack(elements=[EnergyFilter, FlatDetector], keywords=[{'filterfunc': lambda x: 0.5}]),
          ]

# Each elements will be used multiple times.
# Can I add a test to check that using them does not leave any
# extra attributes etc., but that they come out in a clean state?
# How would I do that?


# Make a test photon list
# Some of this should be separate tests, e.g. source position vs. pointing.
# Can I vary energy for e.g. grating?
mysource = PointSource(coords=SkyCoord(30., 30., unit='deg'), energy=1., flux=300.)
masterphotons = mysource.generate_photons(11)
mypointing = FixedPointing(coords=SkyCoord(30., 30., unit='deg'))
masterphotons = mypointing(masterphotons)
myslit = RectangleAperture()
masterphotons = myslit(masterphotons)


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
    (that has to happen in individual tests), but it's a start.
    '''
    def test_one_vs_many_in_call_signature(self, photons, elem):
        '''processing a single photon should give same result as photon list'''
        assert np.all(photons['dir'] == photons['dir'])
        p = photons[[0]]
        # Process individually
        single = photons[0]
        if hasattr(elem, 'process_photon'):
            try:
                # For apertures input pos is ignored, but still needs to be there
                # to keep the function signature consistent.
                dir, pos, energy, pol, prob = elem.process_photon(single['dir'], single['pos'], single['energy'], single['polarization'])
                compare_results = True
            except NotImplementedError:
                # It's OK if only the a vectorized process_photons is implemented
                # but if both exist, they need to return the same answer.
                compare_results = False
        else:
            compare_results = False
        # Process as table
        if isinstance(elem, BaseAperture):
            photons.remove_column('pos')
        p = elem(photons)

        if compare_results:
            # We can only compare, if one and many did run.
            assert np.all(dir == p['dir'][0])
            assert np.all(pos == p['pos'][0])
            assert energy == p['energy'][0]
            assert (np.all(pol == p['polarization'][0]) or
                    (np.all(np.isnan(pol)) and np.all(np.isnan(p['polarization'][0]))))
            assert prob == p['probability'][0]

    def test_desc(self, elem):
        '''every elements should return a Ordered Dict for description'''
        des = elem.describe()
        assert isinstance(des, OrderedDict)
        assert len(des) > 0

    def test_prob(self, photons, elem):
        '''For every type of element, the value for 'probability' can only go down.'''
        if isinstance(elem, BaseAperture):
            photons.remove_column('pos')

        photons['probability'] = 1e-4
        photons = elem(photons)
        assert np.all(photons['probability'] <= 1e-4)

    def test_polarization_direction(self, photons, elem):
        '''For every type of element, the polarization vector must still be
        perpendicular to the direction of the photon after it passed through
        the element. Most elements do not change the direction of the photon,
        so this test is not really relevant, but testing the full list of
        elements here is the only way in make sure that this test is not
        forgotten when individual elements are implemented.'''
        if isinstance(elem, BaseAperture):
            photons.remove_column('pos')

        photons = elem(photons)
        assert np.allclose(np.einsum('...i,...i', photons['dir'].data,
                                     photons['polarization'].data), 0.)
        # Also, the polarization vector must still be a unit vector.
        assert np.allclose(np.linalg.norm(photons['polarization'], axis=1), 1)


    def test_metadata_evolution(self, photons, elem):
        '''In almost all cases, the metadata passed into an element should
        be returned unchanged. New keys can be added all the time, but it is
        extremely rare that an element changes an existing meta-data value.
        (Those cases need to be marked xfail by hand here or be listed in an
        exclusion list when they come up.)

        Regression test for #88.
        '''
        meta_in = copy.deepcopy(photons.meta)
        if isinstance(elem, BaseAperture):
            photons.remove_column('pos')
        p = elem(photons)
        assert all(item in p.meta.items() for item in meta_in.items())

def test_parse_position_keywords_zoom_dimension():
    '''test proper error messages for zoom keyword'''
    for zoom in [np.ones(1), np.ones(2), np.ones(4)]:
        with pytest.raises(ValueError) as e:
            pos4d = _parse_position_keywords({'zoom': zoom})
            assert 'must has three elements' in str(e.value)
