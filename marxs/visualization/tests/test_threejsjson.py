import numpy as np

from ...optics import FlatDetector, RectangleAperture
from ...design import RowlandTorus
from ..threejsjson import plot_rays

def test_box():
    '''Any box shaped element has the same representation so it's
    OK to test just one of them.'''
    det = FlatDetector(zoom=2, position=[2., 3., 4.])
    det.display = det.display
    det.display['opacity'] = 0.1
    out = det.plot(format='threejsjson')
    out_expected = {'geometry': 'BoxGeometry',
                    'geometrypars': (2, 2, 2),
                    'material': 'MeshStandardMaterial',
                    'materialproperties': {'color': '#ffff00',
                                           'opacity': 0.1,
                                           'side': 2,
                                           'transparent': 'true'},
                    'n': 1,
                    'name': "<class 'marxs.optics.detector.FlatDetector'>",
                    'pos4d': [[ 2.,  0.,  0.,  2.,  0.,  2.,  0.,  3.,
                                      0.,  0.,  2.,  4.,  0., 0.,  0.,  1.]]
    }

    # out_expected is only a subset
    for k in out_expected:
        assert out[k] == out_expected[k]


def test_rowland():
    '''Output of Rowland torus'''
    rowland = RowlandTorus(r=2, R=1)
    out = rowland.plot(format='threejsjson')
    out_expected = {'geometry': 'ModifiedTorusBufferGeometry',
                    'material': 'MeshStandardMaterial',
                    'materialproperties': {'color': '#ff4ccc',
                                           'opacity': 0.2,
                                           'side': 2,
                                           'transparent': 'true'},
                    'n': 1,
                    'name': "<class 'marxs.design.rowland.RowlandTorus'>",
    }

    for k in out_expected:
        assert out[k] == out_expected[k]

    assert out['pos4d'] == [np.eye(4).flatten().tolist()]


def test_rays():
    '''Just two rays to make sure the format is right.'''
    rays = plot_rays(np.arange(12).reshape(2,2,3), cmap='jet')
    out_expected = {'color': [[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.5, 1.0],
                              [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.5, 1.0]],
                    'geometry': 'BufferGeometry',
                    'geometrytype': 'Line',
                    'material': 'LineBasicMaterial',
                    'materialproperties': {'vertexColors': 2},
                    'n': 2,
                    'name': 'Photon list',
                    'pos': [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]}

    for k in out_expected:
        assert rays[k] == out_expected[k]

def test_aperture():
    '''Any box shaped element has the same representation so it's
    OK to test just one of them.'''
    det = RectangleAperture(zoom=5)
    det.display = det.display
    det.display['opacity'] = 0.3
    out = det.plot(format='threejsjson')
    out_expected = {'faces': [[0, 4, 5, 0, 1, 5, 1, 5, 6, 1, 2, 6, 2, 6, 7, 2, 3, 7,
                               3, 7, 4, 3, 0, 4]],
                    'geometry': 'BufferGeometry',
                    'geometrytype': 'Mesh',
                    'material': 'MeshStandardMaterial',
                    'materialproperties': {'color': '#00bfff',
                                           'opacity': 0.3,
                                           'side': 2,
                                           'transparent': 'true'},
                    'n': 1,
                    'name': "<class 'marxs.optics.aperture.RectangleAperture'>",
                    'pos': [[0.0, 15.0, 15.0, 0.0, -15.0, 15.0, 0.0, -15.0, -15.0,
                             0.0, 15.0, -15.0, 0.0, 5.0, 5.0, 0.0, -5.0, 5.0, 0.0,
                             -5.0, -5.0, 0.0, 5.0, -5.0]]}

    # out_expected is only a subset
    for k in out_expected:
        assert out[k] == out_expected[k]
