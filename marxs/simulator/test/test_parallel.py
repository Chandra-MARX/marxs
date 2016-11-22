import numpy as np

from ...utils import generate_test_photons
from .. import Parallel
from ...optics import FlatDetector as CCD
from ...design import RowlandTorus, RowlandCircleArray

def test_Parallel_numbering():
    '''Test automatic numbering and numbers assigned to photons.
    '''
    pos = [[0, -10.1, -10.1],[0, .1, -10.1],[0, -10.1, .1],[0, .1, .1]]
    detect = Parallel(elem_class=CCD, elem_args={'pixsize': 0.01, 'zoom': 5},
                      elem_pos={'position': pos}, id_col='CCD_ID')

    for i in range(len(pos)):
        assert detect.elements[i].id_num == i

    photons = generate_test_photons(5)
    photons = detect(photons)
    assert np.all(photons['CCD_ID'] == 3.)

def test_Parallel_offset_numbering():
    '''The randomly generated positions might generate overlapping elements,
    but that is OK to test the numbering scheme.'''
    pos = np.random.rand(5, 3) * 50
    detect = Parallel(elem_class=CCD, elem_args={'pixsize': 0.01, 'zoom': 0.1},
                      elem_pos={'position': pos}, id_num_offset=100)

    for i in range(pos.shape[0]):
        assert detect.elements[i].id_num == i + 100

def test_parallel_calculated_normals():
    '''Regression test: At one point the rotation matrix was reversed,
    mixing active and passive rotations. As a result, the detector elements
    did not match up on the Rowland circle.'''
    rowland = RowlandTorus(5900., 6020.)
    detccdargs = {'pixsize': 0.024,'zoom': [1, 24.576, 12.288]}
    det = RowlandCircleArray(rowland=rowland, elem_class=CCD,
                             elem_args=detccdargs,
                             d_element=49.652, theta=[np.pi - 0.2, np.pi -0.19])
    e1 = det.elements[0]
    e2 = det.elements[1]
    # These two edges should be close together:
    edge1 = e1.geometry('center') + e1.geometry('v_y')
    edge2 = e2.geometry('center') - e2.geometry('v_y')
    assert np.all(np.abs(edge1 - edge2) < 3.)
