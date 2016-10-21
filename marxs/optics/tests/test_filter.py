import numpy as np
import pytest

from ...utils import generate_test_photons
from ..filter import EnergyFilter, GlobalEnergyFilter

@pytest.mark.parametrize("filterclass", [EnergyFilter, GlobalEnergyFilter])
def test_energydependentfilter(filterclass):
    '''Check energy dependent filter'''
    # Useless filter function, but good for testing
    f = filterclass(filterfunc=lambda x: 1./x)
    photons = generate_test_photons(5)
    photons['probability'] = 0.8
    photons['energy'] = np.arange(1., 6.)
    photons = f(photons)
    assert np.allclose(photons['probability'] , 0.8 / np.arange(1., 6.))

@pytest.mark.parametrize("filterclass", [EnergyFilter, GlobalEnergyFilter])
def test_energyfilter_error(filterclass):
    '''Check that Error is raised if probability is > 1 or negative.'''
    photons = generate_test_photons(1)
    for prob in [-0.1, 1.5]:
        f = filterclass(filterfunc=lambda x: np.ones_like(x) * prob)
        with pytest.raises(ValueError) as e:
            temp = f(photons)
        assert 'Probabilities returned by filterfunc' in str(e)

def test_difference_between_global_and_flat_filters():
    '''The global filter does not have a position in space, it just filters all
    photons coming its way. In contrast, photons can pass by the normal filter.'''
    photons = generate_test_photons(5)
    photons['pos'][:, 1] = np.arange(5)

    globalfilter = GlobalEnergyFilter(filterfunc=lambda x: 0.5)
    localfilter = EnergyFilter(filterfunc=lambda x: 0.5, zoom=2.1)

    pg = globalfilter(photons.copy())
    assert np.allclose(pg['probability'] , 0.5)

    pl = localfilter(photons.copy())
    assert set(pl['probability']) == set([0.5, 1.0])
