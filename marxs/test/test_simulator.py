import numpy as np
from astropy.table import Table
import pytest

from ..simulator import Sequence, SimulationSetupError
from ..optics import ThinLens

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
