# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
import pytest

from marxs.missions.mitsnl.catgrating import NonParallelCATGrating
from marxs.missions.mitsnl.tolerances import wiggle_and_bar_tilt

from marxs.design.tests.test_tolerancing import gsa, elempos
from marxs.simulator import SimulationSetupError


def test_change_parallel_elements():
    '''Check that parameters work and elements are in fact changed.
    More detailed checks that the type of change is correct are
    implemented as separate tests, but those tests don't check out
    every parameter.
    '''
    g = gsa(NonParallelCATGrating)
    wiggle_and_bar_tilt(g, 0., 0., 0.)
    assert np.all(np.stack([e.pos4d for e in g.elements]) == elempos)

    for key in ['dx', 'dy', 'dz', 'rx', 'ry', 'rz']:
        d = {key: 1.23}
        wiggle_and_bar_tilt(g, **d)
        assert not np.all(np.stack([e.pos4d for e in g.elements]) == elempos)


def test_wiggle():
    '''Check wiggle function'''
    g = gsa(NonParallelCATGrating)
    wiggle_and_bar_tilt(g, dx=10, dy=.1)
    diff = elempos - np.stack([e.pos4d for e in g.elements])
    # Given the numbers, wiggle in x must be larger than y
    # This also tests that not all diff numbers are the same
    # (as they would be with move).
    assert np.std(diff[:, 0, 3]) > np.std(diff[:, 1, 3])


def test_wiggle_bar_tilt_error():
    '''Check that an error is raised if called on unsuitable object'''
    g = gsa()
    with pytest.raises(SimulationSetupError,
                       match='This wiggle function requires one NonParallelCATGrating per element.'):
        wiggle_and_bar_tilt(g)


def test_wiggle_bar_tilt():
    '''Check wiggle function. Here, we do not wiggle the elements, but just change
    the bar tilt and check that the resulting elements are moved.
    That's not to say they are moved correctly, but at least they are moved.'''
    g = gsa(NonParallelCATGrating)
    wiggle_and_bar_tilt(g, max_bar_tilt=.1)
    diff = elempos - np.stack([e.pos4d for e in g.elements])
    # Given the numbers, wiggle in x must be larger than y
    # This also tests that not all diff numbers are the same
    # (as they would be with move).
    assert np.all(diff[:, 1, :] == pytest.approx(0))
    assert np.all(diff[:, 3, :] == pytest.approx(0))
    # Rotation happens around these axes!
    assert np.all(diff[:, 0, :3] != 0)
    assert np.all(diff[:, 2, :3] != 0)
