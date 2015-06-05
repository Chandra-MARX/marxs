import numpy as np
import astropy.units as u

from .. import energy2wave

def test_energy2wave():
    '''Check units and formula for energy2wave'''
    energies = np.arange(.1, 10., .1)
    np.allclose(energy2wave / energies,
                (energies * u.keV).to(u.mm, equivalencies=u.spectral()).value
                )
