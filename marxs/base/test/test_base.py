# Licensed under GPL version 3 - see LICENSE.rst
import pytest
import numpy as np
from astropy.table import Table

from ..base import _parse_position_keywords
from ..base import SimulationSequenceElement as SSE


def test_invalid_zoom():
    '''This error is easy to make, e.g. that the thickness of
    an aperture to 0, thus is deserves its own test and its own
    error message.
    '''
    with pytest.raises(ValueError) as e:
        _parse_position_keywords({'zoom': [0, 0, 1]})
    assert 'All values in zoom must be positive' in str(e.value)


def test_invalid_pos4d():
    pos4d = np.array([[0,  0, -7.5, -1.35],
                      [0.,  5.5,  0.,  0],
                      [0.,  0.,  4.e-15,  1.4],
                      [0.,  0.,  0.,  1.]])
    with pytest.raises(ValueError) as e:
        _parse_position_keywords({'pos4d': pos4d})
    assert "is invalid" in str(e.value)


def test_add_output_cols():
    '''Test different ways to set up output columns'''
    class T(SSE):
        output_columns = ['a', 'b']

    photons = Table({'a': [1, 2]})
    t = T()
    t.add_output_cols(photons)
    assert photons.colnames == ['a', 'b']
    # An existing column is not reset to initial value
    assert photons['a'][1] == 2
    # Default initial value is nan
    assert np.isnan(photons['b'][1])

    # Add two columns at once, but in different format
    t.add_output_cols(photons, [{'name': '123', 'dtype': int}, 'text'])
    assert np.isnan(photons['text'][1])
    assert np.issubdtype(photons['123'].dtype, np.integer)


def test_add_idnum_col():
    '''Text another mechanism to add output columns'''
    class T(SSE):
        id_col = 'my_id'

    photons = Table({'a': [1, 2]})
    t = T()
    t.add_output_cols(photons)
    assert np.all(photons['my_id'] == -1)
    assert np.issubdtype(photons['my_id'].dtype, np.integer)


def test_output_col_none():
    '''Setting colnames to None, means that colum is not added.'''
    class T(SSE):
        id_col = None

    photons = Table({'a': [1, 2]})
    t = T()
    t.add_output_cols(photons)
    assert photons.colnames == ['a']
