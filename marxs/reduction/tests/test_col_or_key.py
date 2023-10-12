# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from ..ogip import ColOrKeyTable

import pytest

@pytest.mark.parametrize('input', [3, [3, 3], 3 * np.ones(2, dtype=int)])
def test_insert_allsame(input):
    t = ColOrKeyTable([['text1', 'text2'], [1, 2]], names=['a', 'b'])
    t['c'] = input
    assert t.meta['c'] == 3
    assert t['c'] == 3
    assert t.colnames == ['a', 'b']


def test_access_meta_as_column():
    t = ColOrKeyTable([['text1', 'text2'], [1, 2]], names=['a', 'b'])
    t.meta['test'] = 'mytext'
    assert t['test'] == 'mytext'


def test_existing_col_allsame():
    t = ColOrKeyTable([['text1', 'text2'], [1, 2]], names=['a', 'b'])
    t['a'] = 'text3'
    assert t.meta['a'] == 'text3'
    assert t['a'] == 'text3'
    assert t.colnames == ['b']


def test_existing_col_allsame_index():
    t = ColOrKeyTable([['text1', 'text2'], [1, 2]], names=['a', 'b'])
    t['a'][1] = 'text1'
    assert t.meta['a'] == 'text1'
    assert t['a'] == 'text1'
    assert t.colnames == ['b']


def test_existing_col_allsame_indexarray():
    t = ColOrKeyTable([['text1', 'text2', 'text2'], [1, 2, 3]], names=['a', 'b'])
    t['a'][t['a'] == 'text2'] = 'text1'
    assert t.meta['a'] == 'text1'
    assert t['a'] == 'text1'
    assert t.colnames == ['b']

@pytest.mark.skip()
def test_setitem_on_meta_as_column():
    '''Have information that's in meta, but then change one element'''
    t = ColOrKeyTable([['text1', 'text2'], [1, 2]], names=['a', 'b'])
    t.meta['test'] = 'mytext'
    t['test'][1] = 'mytext_different'
    assert t.colnames == ['a', 'b', 'test']
    assert np.all(t['test'] == ['mytext', 'mytext', 'mytext_different'])