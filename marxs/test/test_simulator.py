import numpy as np
from astropy.table import Table
import pytest

from ..simulator import Sequence, SimulationSetupError, Parallel
from ..optics import ThinLens, FlatGrating, uniform_efficiency_factory

def test_pre_post_process():
    '''test pre-processing and post-processing in sequences'''

    def set_energy2(photons):
        photons['energy'] = 2
        return photons

    def set_energy3(photons):
        photons['energy'] = 3
        return photons

    def process(photons):
        photons.meta['mean_en'].append(photons['energy'].mean())

    photons = Table({'energy': [1,1,1]})
    photons.meta['mean_en'] = []

    seq_pre = Sequence(sequence=[set_energy2, set_energy3], preprocess_steps=[process])
    tpre = seq_pre(photons.copy())
    assert np.all(tpre['energy'] == 3)
    assert np.all(tpre.meta['mean_en'] == [1, 2])

    seq_post = Sequence(sequence=[set_energy2, set_energy3], postprocess_steps=[process])
    tpost = seq_post(photons.copy())
    assert np.all(tpost['energy'] == 3)
    assert np.all(tpost.meta['mean_en'] == [2, 3])

def test_badinput():
    '''Inputs must be callable.'''
    with pytest.raises(SimulationSetupError) as e:
        seq = Sequence(sequence=[5])
    assert "is not callable" in str(e.value)

    with pytest.raises(SimulationSetupError) as e:
        seq = Sequence(sequence=[ThinLens], preprocess_steps=[5])
    assert "is not callable" in str(e.value)

# Tests for Parallel could be here, but Parallel is a fairly general container.
# It is easier to test a specific case - and the tests for GratingArrayStructure do that
# for many functions or Parallel.

def test_list_elemargs():
    '''Input to elemargs can be single or list.'''
    p0 = Parallel(elem_class=FlatGrating,
                  elem_pos={'position': [np.zeros(3), np.ones(3)]},
                  elem_args={'order_selector': uniform_efficiency_factory(), 'd': [0.001, 0.002], 'zoom': [3,3]},
                 )
    # equivalent way of inputting things
    p1 = Parallel(elem_class=FlatGrating,
                  elem_pos={'position': [np.zeros(3), np.ones(3)], 'zoom': [3, 3]},
                  elem_args={'order_selector': uniform_efficiency_factory(), 'd': [0.001, 0.002]},
                 )
    # yet another way
    p2 = Parallel(elem_class=FlatGrating,
                  elem_pos={'position': [np.zeros(3), np.ones(3)]},
                  elem_args={'order_selector': uniform_efficiency_factory(), 'd': [0.001, 0.002], 'zoom': 3},
                  )
    for p in [p0, p1, p2]:
        assert len(p.elements) == 2
        for e in p.elements:
            assert np.linalg.norm(e.geometry['v_y']) == 3
        assert p.elements[0].d == 0.001
        assert p.elements[1].d == 0.002

    # but this does not work: in elem_pos all entries have to be lists
    with pytest.raises(ValueError) as e:
        p4 = Parallel(elem_class=FlatGrating,
                      elem_pos={'position': [np.zeros(3), np.ones(3)], 'zoom': 3},
                      elem_args={'order_selector': uniform_efficiency_factory(), 'd': [0.001, 0.002]},
                      )
    assert 'All elements in elem_pos must have the same number' in str(e.value)

