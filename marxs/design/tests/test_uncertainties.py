from copy import deepcopy
import numpy as np

from ..rowland import RowlandTorus, GratingArrayStructure
from ..uncertainties import generate_facet_uncertainty
from ...optics import FlatGrating, constant_order_factory

def test_uncertainty_generation():
    '''The best way to test that the output format is reasonable is to use it.

    I do not really have an idea how I test that the distribution is right without
    doing the exact reverse (fit the distribution) of the procedure we are
    testing here.
    Also, generating the GAS is a fairly expensive operation, so I do not want to
    make too many elements unless there is a good reason for it.
    '''

    np.random.seed(10)

    rowland  = RowlandTorus(5e3, 5e3)
    gas = GratingArrayStructure(rowland, d_element=2., x_range=[9e3, 1.1e4],
                                radius=[2, 15], phi=[0, np.pi],
                                elem_class=FlatGrating,
                                elem_args={'d': 2e-4,
                                           'order_selector': constant_order_factory(1)},
    )
    oldgaspos = deepcopy(gas.elem_pos)
    for a, b in zip(oldgaspos, gas.elements):
        # Check position record in GAS and in elements is in snyc
        assert np.allclose(a, b.pos4d)

    # Note that input is list, which will be converted to ndarray internally
    gas.elem_uncertainty = generate_facet_uncertainty(len(gas.elem_uncertainty),
                                                      [0, 0, 1.], np.zeros(3))
    # Now apply the uncertainties to make sure it works
    gas.generate_elements()

    for a, b in zip(gas.elem_pos, gas.elements):
        # Check position record in GAS and in elements is still in snyc
        assert ~np.allclose(a, b.pos4d)

    # Check that the shifts are right
    shifts = np.array([e[2, 3] - f.pos4d[2, 3] for e, f in zip(oldgaspos, gas.elements)])
    assert np.isclose(np.mean(shifts), 0., atol=0.1)
    assert np.isclose(np.std(shifts), 1., atol=0.1)
