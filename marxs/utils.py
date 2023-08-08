import numpy as np
from astropy.table import Table
import warnings

__all__ = ['SimulationSetupWarning', 'generate_test_photons',
           'DataFileFormatException', 'tablerows_to_2d']


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
    - polarization: y-direction

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


class DataFileFormatException(Exception):
    '''Exception for grating efficiency files not matching expected format.'''
    pass


def tablerows_to_2d(tab):
    '''Get a 2d array from an input table.

    In the table, the data is flattened to a 1d form.
    The first two columns are x and y, like this:
    The first column looks like this with many duplicates:
    [1,1,1,1,1,1,2,2,2,2,2,2,3,3,3, ...].
    Column B repeats like this: [1,2,3,4,5,6,1,2,3,4,5,6,1,2,3, ...].

    All remaining columns are data on the same x-y grid, and the grid
    has to be regular.

    Parameters
    ----------
    tab : `astropy.table.Table`
        Table as read in. Useful to access units or other meta data.

    Returns
    -------
    x, y : `astropy.table.Column`
        Unique entries in first and second column
    dat : np.array
        The remaining outputs are np.arrays of shape (len(x), len(y))
    '''
    x = tab.columns[0]
    y = tab.columns[1]
    n_x = len(set(x))
    n_y = len(set(y))
    if len(x) != (n_x * n_y):
        raise DataFileFormatException('Data is not on regular grid.')

    x_arr = tab.columns[0].data.reshape(n_x, n_y)
    y_arr = tab.columns[1].data.reshape(n_x, n_y)
    if not (np.allclose(x_arr, x_arr[:, 0][:, None]) and
            np.allclose(y_arr, y_arr[0, :][None, :])):
        raise DataFileFormatException('Input table x, y not sorted as expected.')

    x = x[::n_y]
    y = y[:n_y]
    coldat = [tab[d].data.reshape(n_x, n_y) for d in tab.columns[2:]]

    return x, y, coldat
