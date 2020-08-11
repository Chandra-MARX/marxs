# Licensed under GPL version 3 - see LICENSE.rst
import pytest

from ..base import _parse_position_keywords


def test_invalid_zoom():
    '''This error is easy to make, e.g. that the thickness of
    an aperture to 0, thus is deserves its own test and its own
    error message.
    '''
    with pytest.raises(ValueError) as e:
        _parse_position_keywords(zoom=[ 0, 0, 1])
    assert 'All values in zoom must be positive' in str(e.value)


def test_invalid_pos4d():
    pos4d = np.array([[ 0,  0, -7.5, -1.35],
                      [ 0.,  5.5,  0.,  0],
                      [ 0.,  0.,  4.e-15,  1.4],
                      [ 0.,  0.,  0.,  1.]])
    with pytest.raises(ValueError) as e:
        _parse_position_keywords(pos4d=pos4d)
    assert "is invalid" in str(e.value)
