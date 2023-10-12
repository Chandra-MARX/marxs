# Licensed under GPL version 3 - see LICENSE.rst
import os
import logging

from astropy.table import Table
import astropy.units as u
from .utils import config, git_hash, git_info

hash_displayed = False


class DataFileFormatException(Exception):
    pass


def log_tab_metadata(dirname, filename):
    '''Print information about loaded files to standard out.

    Parameters
    ----------
    dirname : string
        Name for the directory in the caldb-input file structure
    filename : string
        Name of data file (without the ".csv" part)
    '''
    global hash_displayed
    if not hash_displayed:
        text = 'data files in {}: git hash {} (committed on {})'
        logging.info(text.format(config['data']['caldb_inputdata'],
                                 git_hash, git_info))
        hash_displayed = True
    logging.info('Loading data from {0}/{1}'.format(dirname, filename))


def load_number(dirname, filename, valuename):
    '''Get a single number from an ecsv input file

    Parameters
    ----------
    dirname : string
        Name for the directory in the caldb-input file structure
    filename : string
        Name of data file (without the ".csv" part)
    valuename : string
        Name of the column that hold the value

    Returns
    -------
    val : float or `astropy.units.Quantity`
        If the unit of the column is set, returns a `astropy.units.Quantity`
        instance, otherwise a plain float.
    '''
    tab = Table.read(os.path.join(config['data']['caldb_inputdata'], dirname,
                                  filename + '.csv'), format='ascii.ecsv')
    log_tab_metadata(dirname, filename)
    if len(tab) != 1:
        raise DataFileFormatException('Table {} contains more than one row of data.'.format(filename))
    else:
        if tab[valuename].unit is None:
            return tab[valuename][0]
        else:
            return u.Quantity(tab[valuename])[0]


def load_table(dirname, filename):
    '''Get a table from an ecsv input file

    Parameters
    ----------
    dirname : string
        Name for the directory in the caldb-input file structure
    filename : string
        Name of data file (without the ".csv" part)

    Returns
    -------
    val : `astropy.table.Table`
    '''
    tab = Table.read(os.path.join(config['data']['caldb_inputdata'], dirname,
                                  filename + '.csv'), format='ascii.ecsv')
    log_tab_metadata(dirname, filename)
    return (tab)
