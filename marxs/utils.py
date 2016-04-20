import numpy as np
from astropy.table import Table

def generate_test_photons(n=1):
    '''Generate a photon structure for testing.

    This function returns a table of ``n`` identical photons with
    standard properties:

    - position: ``x=1; y=0; z=0``
    - direction: negative x-axis
    - energy: 1 keV
    - polarization: 1

    This is useful for testing purposes.

    Parameters
    ----------
    n : int
        number of photons to generate

    Returns
    -------
    photons : `astropy.table.Table`
        Table of ``n`` identical photons.
    '''
    dir = np.tile(np.array([-1., 0., 0., 0.]), (n, 1))
    pos = np.tile(np.array([1., 0., 0., 1.]), (n, 1))
    photons = Table({'pos': pos,
                     'dir': dir,
                     'energy': np.ones(n),
                     'polarization': np.ones(n),
                     'probability': np.ones(n),
                     })
    return photons
