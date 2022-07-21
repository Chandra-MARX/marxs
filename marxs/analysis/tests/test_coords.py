# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
import pytest

from marxs.analysis.coords import ProjectOntoPlane, facet_table
from marxs.optics.base import OpticalElement
from marxs.math.utils import xyz2zxy
from marxs.simulator import Parallel
from marxs.analysis import ProjectOntoPlane


def test_facet_table():
    '''Make a facet table. At the same time, this tests ProjectOntoPlane'''
    gas = Parallel(elem_class=OpticalElement, elem_pos={'position':[[1, 1, 1], [2, 0, 4]]})
    ftab = facet_table(gas)
    proj2 = ProjectOntoPlane(orientation=xyz2zxy[:3, :3])
    ftab2 = facet_table(gas, proj2)
    assert ftab.colnames == ['facet_projx','facet_projy', 'facet_ang', 'facet_rad',
                             'facet', 'facet_x', 'facet_y', 'facet_z']
    assert len(ftab) == 2
    assert np.all(ftab['facet_projy'] == [1, 4])
    assert np.all(ftab2['facet_projy'] == [1, 0])
    assert np.all(ftab['facet_ang'] == pytest.approx([np.pi / 4, np.pi / 2]))
    assert np.all(ftab2['facet_ang'] == pytest.approx([np.pi / 4, 0]))
    assert dict(ftab2[1]) == {'facet_projx': 2.0,
                              'facet_projy': 0.0,
                              'facet_ang': 0.0,
                              'facet_rad': 2.0,
                              'facet': 1,
                              'facet_x': 2.0,
                              'facet_y': 0.0,
                              'facet_z': 4.0}
