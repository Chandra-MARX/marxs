import numpy as np
import pytest
from astropy.table import Table

from ..source import Source, SourceSpecificationError

def test_energy_input_default():
    '''For convenience and testing, defaults for time, energy and pol are set.'''
    s = Source()
    photons = s.generate_photons(5.1)
    assert len(photons) == 5
    assert np.all(photons['energy'] == 1.)
    assert len(set(photons['pol_angle'])) == 5  # all pol angles are different

def test_flux_input():
    '''Options: contant rate or function'''
    # 1. constant rate
    s = Source(flux=5)
    photons = s.generate_photons(5.01)
    assert len(photons) == 25
    delta_t = np.diff(photons['time'])
    assert np.all(delta_t == delta_t[0]) # constante rate

    # 2. function
    def f(t):
        return np.logspace(1, np.log10(t))

    s = Source(flux=f)
    photons = s.generate_photons(100)
    assert np.all(photons['time'] == np.logspace(1, 2))

    # 3. anything else
    s = Source(flux=ValueError)
    with pytest.raises(SourceSpecificationError) as e:
        photons = s.generate_photons(5)
    assert '`flux` must be' in str(e.value)

def test_energy_input():
    '''Many differnet options ...'''
    # 1. contant energy
    s = Source(energy=2.)
    photons = s.generate_photons(5)
    assert np.all(photons['energy'] == 2.)

    # 2. function
    def f(t):
        return t
    s = Source(energy=f)
    photons = s.generate_photons(5)
    assert np.all(photons['energy'] == photons['time'])
    # bad function
    s = Source(energy=np.sum)
    with pytest.raises(SourceSpecificationError) as e:
        photons = s.generate_photons(5)
    assert 'an array of same size as' in str(e.value)

    # 3. array, table, recarray, dict, ... just try some of of the opion with keys
    # spectrum with distinct lines
    engrid = [0.5, 1., 2., 3.]
    fluxgrid = [456., 1., 0., 2.]  # first entry (456) will be ignored
    s1 = Source(energy={'energy': engrid, 'flux': fluxgrid})
    s2 = Source(energy=np.vstack([engrid, fluxgrid]))
    s3 = Source(energy=Table({'energy': engrid, 'flux': fluxgrid}))
    for s in [s1, s2, s3]:
        photons = s.generate_photons(1000)
        en = photons['energy']
        ind0510 = (en >= 0.5) & (en <=1.0)
        ind2030 = (en >= 2.) & (en <=3.0)
        assert (ind0510.sum() + ind2030.sum()) == len(photons)
        assert ind0510.sum() < ind3030.sum()

    # 4. anything else
    s = Source(energy=ValueError)
    with pytest.raises(SourceSpecificationError) as e:
        photons = s.generate_photons(5)
    assert '`energy` must be' in str(e.value)

def test_polarization_input():
    '''Many differnet options ...'''
    # 1. 100 % polarized flux
    s = Source(polarization=2.)
    photons = s.generate_photons(5)
    assert np.all(photons['plo_angle'] == 2.)

    # 2. function
    def f(t, en):
        return t * en
    s = Source(polarization=f, energy=2.)
    photons = s.generate_photons(5)
    assert np.all(photons['pol_angle'] == photons['time'] * photons['energy'])
    # bad function
    s = Source(ploarization=np.dot)
    with pytest.raises(SourceSpecificationError) as e:
        photons = s.generate_photons(5)
    assert 'an array of same size as' in str(e.value)

    # 3. array, table, recarray, dict, ... just try some of of the opion with keys
    # spectrum with distinct lines
    polgrid = [0.5, 1., 2., 3.]
    probgrid = [456., 1., 0., 2.]  # first entry (456) will be ignored
    s1 = Source(ploarization={'angle': polgrid, 'probability': probgrid})
    s2 = Source(polarization=np.vstack([polgrid, probgrid]))
    s3 = Source(polarization=Table({'angle': polgrid, 'probability': probgrid}))
    for s in [s1, s2, s3]:
        photons = s.generate_photons(1000)
        pol = photons['pol_angle']
        ind0510 = (en >= 0.5) & (en <=1.0)
        ind2030 = (en >= 2.) & (en <=3.0)
        assert (ind0510.sum() + ind2030.sum()) == len(photons)
        assert ind0510.sum() < ind3030.sum()

    # 4. None (unpolarized source)
    s = Source(polarization=None)
    photons = s.generate_photons(5)
    assert len(set(photons['pol_angle'])) == len(photons) # all different

    # 5. anything else
    s = Source(energy=ValueError)
    with pytest.raises(SourceSpecificationError) as e:
        photons = s.generate_photons(5)
    assert '`polarizarion` must be' in str(e.value)
