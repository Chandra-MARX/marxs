# Licensed under GPL version 3 - see LICENSE.rst
from astropy.table import Table
import numpy as np
import pytest

from marxs.utils import DataFileFormatException, tablerows_to_2d

def test_tab_2d_OK():
    '''read a 3 * 2 table'''
    tab = Table(rows=[[1., 1., 0], [1., 2., 1],
                      [1.5, 1., 2], [1.5, 2., 3],
                      [1.8, 1., 4], [1.8, 2., 5]],
                names=['x', 'y','dat'])
    x, y, cols = tablerows_to_2d(tab)
    assert np.all(x == [1, 1.5, 1.8])
    assert np.all(y == [1, 2])
    assert np.all(cols[0] == np.arange(6).reshape((3, 2)))
    
    
def test_tab_2d_sizeerror():
    '''Like the first example, but the second row is missing.'''
    tab = Table(rows=[[1, 1, 1],
                      [1.5, 1., 3], [1.5, 2., 4.],
                      [1.8, 1., 5], [1.8, 2., 6.]],
                names=['x', 'y','dat'])
    with pytest.raises(DataFileFormatException, match='Data is not'):
        x, y, cols = tablerows_to_2d(tab)
            
    
def test_tab_2d_sorterror():
    '''Like the first example, but first and second entry are exchanged.'''
    tab = Table(rows=[[1, 2, 2],  [1, 1, 1],
                      [1.5, 1., 3], [1.5, 2., 4.],
                      [1.8, 1., 5], [1.8, 2., 6.]],
                names=['x', 'y','dat'])
    with pytest.raises(DataFileFormatException, match='not sorted'):
        x, y, cols = tablerows_to_2d(tab)