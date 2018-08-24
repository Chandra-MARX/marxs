import numpy as np
from astropy.table import Table
import pytest

from ..simulator import KeepCol, Sequence, Parallel, BaseContainer
from ...optics import FlatDetector

f1 = FlatDetector()
f2 = FlatDetector()
f3 = FlatDetector()
s_l2 = Sequence(elements=[f2, f3])

mission = Sequence(elements=[f1, s_l2])

def test_seach_all():
    '''Check full search finding all elements.'''
    assert [mission, s_l2] == mission.elements_of_class(Sequence)
    assert [] == mission.elements_of_class(BaseContainer)
    assert [mission, s_l2] == mission.elements_of_class(BaseContainer,
                                                        subclass_ok=True)
    assert [f1, f2, f3] == mission.elements_of_class(FlatDetector)
    assert [f1, f2, f3] == mission.elements_of_class(FlatDetector, subclass_ok=True)

def test_seach_top():
    '''Check top search finding all elements.'''
    assert [mission] == mission.elements_of_class(Sequence, stop_at_first=True)
    assert [] == mission.elements_of_class(BaseContainer, stop_at_first=True)
    assert [mission] == mission.elements_of_class(BaseContainer,
                                                  subclass_ok=True,
                                                  stop_at_first=True)
    assert [f1, f2, f3] == mission.elements_of_class(FlatDetector,
                                                     stop_at_first=True)
    assert [f1, f2, f3] == mission.elements_of_class(FlatDetector,
                                                     subclass_ok=True,
                                                     stop_at_first=True)


def test_format_saved_positions():
    '''Reformat saved positions and drop nearly identical values.'''
    pos0 = np.arange(20).reshape(5,4)
    pos1 = pos0 + 1
    pos2 = pos1 + 1e-4
    pos = KeepCol('testcase')
    pos.data = [pos0, pos1, pos2]

    d = pos.format_positions(atol=1e-2)
    assert d.shape == (5, 2, 3)
    assert np.allclose(d[0, 0, :], [0, 1./3, 2./3])

    '''Not drop values'''
    d = pos.format_positions()
    assert d.shape == (5, 3, 3)

def test_empty_format_saved_positions():
    '''If the input contains no data, an error should be raised.'''
    a = KeepCol('testcase')
    with pytest.raises(ValueError) as e:
        d = a.format_positions()

    assert 'contains no data' in str(e.value)

def test_format_save_positions_fails():
    '''Can only be used when data is in homogenous coordiantes.'''
    a = KeepCol('testcase')
    a.data = [np.arange(5), np.arange(5)]
    with pytest.raises(ValueError) as e:
        d = a.format_positions()

    assert 'homogeneous coordinates' in str(e.value)

def test_to_array():
    a = KeepCol('testcase')
    col1 = Table([[1]], names=['testcase'])
    col2 = Table([[4]], names=['testcase'])
    a(col1)
    a(col2)
    assert np.all(np.sqrt(a) == [[1.], [2.]])
