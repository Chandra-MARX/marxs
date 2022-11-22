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

