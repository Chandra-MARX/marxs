import numpy as np
import pytest

from ..simulator import KeepCol

def test_format_saved_positions():
    '''Reformat saved positions and drop nearly identical values.'''
    pos0 = np.arange(20).reshape(5,4)
    pos1 = pos0 + 1
    pos2 = pos1 + 1e-4
    pos = KeepCol('testcase')
    pos.data = [pos0, pos1, pos2]

    d = pos.to_array(atol=1e-2)
    assert d.shape == (5, 2, 3)
    assert np.allclose(d[0, 0, :], [0, 1./3, 2./3])

    '''Not drop values'''
    d = pos.to_array()
    assert d.shape == (5, 3, 3)

def test_empty_format_saved_positions():
    '''If the input contains no data, an error should be raised.'''
    a = KeepCol('testcase')
    with pytest.raises(ValueError) as e:
        d = a.to_array()

    assert 'contains no data' in str(e.value)
