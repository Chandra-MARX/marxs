# Licensed under GPL version 3 - see LICENSE.rst
import os
import logging
import subprocess

from astropy.table import Table
import astropy.units as u
from .utils import config



# subprocess uses fork internally, which makes a child process
# that essentially makes a copy of everything that python has
# in memory, essentially doubling the memory footprint of python
# at that point. For long running scripts with big simulations that
# can be enough to run out of memory.
# So, just get this info once during the first import and save in
# module-level variables.
git_hash = subprocess.check_output(["git", "describe", "--always"],
                                   cwd=config['data']['caldb_inputdata'])[:-1]

git_info = subprocess.check_output(['git', 'show', '-s', '--format=%ci',
                                    git_hash],
                                   cwd=config['data']['caldb_inputdata'])

hash_displayed = False


class DataFileFormatException(Exception):
    pass


def log_tab_metadata(dirname, filename, tab):
    '''Print information about loaded files to standard out.

    Parameters
    ----------
    dirname : string
        Name for the directory in the caldb-input file structure
    filename : string
        Name of data file (without the ".csv" part)
    valuename : string
        Name of the column that hold the value
    '''
    global hash_displayed
    if not hash_displayed:
        text = 'data files in {}: git hash {} (commited on {})'
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
    log_tab_metadata(dirname, filename, tab)
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
    log_tab_metadata(dirname, filename, tab)
    return (tab)
