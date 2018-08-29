import numpy as np
import pytest

from ..tolerancing import (oneormoreelements,
                           wiggle, moveglobal, moveindividual,
                           varyperiod, varyorderselector, varyscatter,
                           )
from ...optics import FlatGrating, OrderSelector
from ...design import RowlandTorus, GratingArrayStructure


mytorus = RowlandTorus(0.5, 0.5, position=[1.5, 0, -3])

def gsa():
    '''make a parallel structure - fresh for every test'''
    g = GratingArrayStructure(mytorus, d_element=0.1, x_range=[0.5, 1.], radius=[0.1,.2],
                              elem_class=FlatGrating,
                              elem_args={'zoom':0.2, 'd':0.002,
                                         'order_selector': OrderSelector([1])
                                     })
    return g

elempos = np.stack(gsa().elem_pos)

def test_oneormore():
    @oneormoreelements
    def func(a, b, c):
        a.value += 1

    class HoldData():
        def __init__(self, value):
            self.value = value

    obj1 = HoldData(2)
    obj2 = HoldData(4)
    obj3 = HoldData(6)

    listin = [obj2, obj3]

    # First, make sure that func works, otherwise the remaining test is useless.
    func(obj1, 2, c=4)
    assert obj1.value == 3

    func(listin, 'a', None)
    assert listin[0].value == 5
    assert listin[1].value == 7


@pytest.mark.parametrize('function', [wiggle, moveglobal, moveindividual])
def test_change_parallel_elements(function):
    '''Check that parameters work and elements are in fact changed.
    More detailed checks that the type of change is correct are
    implemented as separate tests, but those tests don't check out
    every parameter/
    '''
    g = gsa()
    function(g, 0., 0., 0.)
    assert np.all(np.stack([e.pos4d for e in g.elements]) == elempos)

    for key in ['dx', 'dy', 'dz', 'rx', 'ry', 'rz']:
        d = {key: 1.23}
        function(g, **d)
        assert not np.all(np.stack([e.pos4d for e in g.elements]) == elem_pos)


def test_moveelements_translate():
    '''Check that the element movers work. If the whole structure is translated
    or individual elements are translated by the same amount, the positions
    should be the same.'''
    g1 = gsa()
    g2 = gsa()
    moveglobal(g1, dy=-20)
    moveindividual(g2, dy=-20)
    assert np.allclose(np.stack([e.pos4d for e in g1.elements]),
                       np.stack([e.pos4d for e in g2.elements]))


def test_moveelements_rotate():
    '''Check that the element movers work.
    Unlike test_moveelements_translate we expect different results because
    there are different center of the rotation.
    This test does not check that the rotation is correct, only that its
    different because the validity of the rotation matrix itself is already
    covered by the tests in the transforms3d package.
    '''
    g1 = gsa()
    g2 = gsa()
    moveglobal(g1, rz=-1, ry=.2)
    moveindividual(g2, rz=-1, ry=.2)
    assert not np.allclose(np.stack([e.pos4d for e in g1.elements]),
                           np.stack([e.pos4d for e in g2.elements]))


def test_wiggle():
    '''Check wiggle function'''
    g = gsa()
    wiggle(g, dx=10, dy=.1)
    diff = elempos - np.stack([e.pos4d for e in g.elements])
    # Given the numbers, wiggle in x must be larger than y
    # This also tests that not all diff number are the same
    # (as they would be with move).
    assert np.std(diff[:, 0, 0]) > np.std(diff[:, 0, 1])


@pytest.mark.parametrize('function', [varyperiod, varyorderselector, varyscatter])
def test_errormessage(function):
    '''Check that check is performed for right type of object.
    Some function just set an attribute and there is no function call after
    that that would fail or do anything if called with the wrong type of object.
    Thus, it's very simple to call these with an object where it does not make
    any sense to apply them. SO, they have some error check. Here, we check
    this check.
    '''
    with pytest.raises(ValueError) as e:
        # All functions accept two parameters.
        # Error should be raised before they are used, so the value does not
        # matter
        function(gsa, 1., 2.)

    assert 'does not have' in str(e.value)


def test_gratings_d():
    '''Change the grating constant.'''
    g = gsa()
    varyperiod(g.elements, 1., .1)
    periods = [e._d for e in g.elements]
    assert np.std(periods) > 0.01
    assert np.std(periods) < 5.
    assert np.mean(periods) > .5
