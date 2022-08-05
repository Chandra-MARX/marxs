from astropy.table import Table
import astropy.units as u


class DataFileFormatException(Exception):
    pass


def load_number(filename, valuename):
    '''Get a single number from an ecsv input file

    Parameters
    ----------
    filename : string
        Name of data file
    valuename : string
        Name of the column that hold the value

    Returns
    -------
    val : float or `astropy.units.Quantity`
        If the unit of the column is set, returns a `astropy.units.Quantity`
        instance, otherwise a plain float.
    '''
    tab = Table.read(filename, format='ascii.ecsv')
    if len(tab) != 1:
        raise DataFileFormatException('Table {} contains more than one row of data.'.format(filename))
    else:
        if tab[valuename].unit is None:
            return tab[valuename][0]
        else:
            return u.Quantity(tab[valuename])[0]


def load_table(filename):
    '''Get a table from an ecsv input file

    Parameters
    ----------
    filename : string
        Name of data file

    Returns
    -------
    val : `astropy.table.Table`
    '''
    tab = Table.read(filename, format='ascii.ecsv')
    return (tab)


def load_table2d(filename):
    '''Get a 2d array from an ecsv input file.

    In the table file, the data is flattened to a 1d form.
    The first two columns are x and y, like this:
    The first column looks like this with many duplicates:
    [1,1,1,1,1,1,2,2,2,2,2,2,3,3,3, ...].
    Column B repeats like this: [1,2,3,4,5,6,1,2,3,4,5,6,1,2,3, ...].

    All remaining columns are data on the same x-y grid, and the grid
    has to be regular.


    Parameters
    ----------
    filename : string
        Name of data file

    Returns
    -------
    x, y : `astropy.table.Column`
    colnames : list
        List of names of the other columns (which hold the data)
    dat : np.array
        The remaining outputs are np.arrays of shape (len(x), len(y))
    '''
    tab = Table.read(filename, format='ascii.ecsv')

    x = tab.columns[0]
    y = tab.columns[1]
    n_x = len(set(x))
    n_y = len(set(y))
    if len(x) != (n_x * n_y):
        raise DataFileFormatException('Data is not on regular grid.')

    x = x[::n_y]
    y = y[:n_y]
    colnames = tab.colnames[2:]
    coldat = [tab[d].data.reshape(n_x, n_y) for d in tab.columns[2:]]

    return x, y, colnames, coldat
