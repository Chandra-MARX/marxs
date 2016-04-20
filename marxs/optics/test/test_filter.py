import numpy as np
import pytest

from ...utils import generate_test_photons
from ..filter import EnergyFilter

def test_energydependentfilter():
    '''Check energy dependent filter'''
    # Useless filte rfunction, but good for testing
    f = EnergyFilter(filterfunc=lambda x: 1./x)
    photons = generate_test_photons(5)
    photons['energy'] = np.arange(1., 6.)
    photons = f(photons)
    assert np.allclose(photons['probability'] , 1./np.arange(1., 6.))


def test_energyfilter_error():
    '''Check that Error is raised if probability is > 1 or negative.'''
    photons = generate_test_photons(1)
    for prob in [-0.1, 1.5]:
        f = EnergyFilter(filterfunc=lambda x: np.ones_like(x) * prob)
        with pytest.raises(ValueError) as e:
            temp = f(photons)
        assert 'Probabilities returned by filterfunc' in str(e)
