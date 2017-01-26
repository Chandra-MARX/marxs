import numpy as np
from astropy.table import Table
from astropy.utils.metadata import MergeStrategy
import warnings

class SimulationSetupWarning(Warning):
    pass

warnings.filterwarnings("always", ".*", SimulationSetupWarning)


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
    pol = np.tile(np.array([0., 1., 0., 0.]), (n, 1))
    photons = Table({'pos': pos,
                     'dir': dir,
                     'energy': np.ones(n),
                     'polarization': pol,
                     'probability': np.ones(n),
                     })
    return photons


class MergeIdentical(MergeStrategy):
    '''Merge metadata in astropy table

    In some cases, a table of photons is split up, e.g. when half of
    the photons passes through one aperture and the other half through
    a second aperture. When merging those tables back together, they
    have the same metadata, but the default `astropy.utils.metadata.MergePlus`
    would lead to doubled entries.
    '''
    types = [(list, list), (tuple, tuple)]

    @classmethod
    def merge(cls, left, right):
        if left == right:
            return left
        else:
            return left + right
