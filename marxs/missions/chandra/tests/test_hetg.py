# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np

from .. import HETG

def test_orient():
    '''Check orientation of gratings.'''
    hetg = HETG()
    for i in range(len(hetg.elements)):
        ex = hetg.elements[i].geometry['e_x'][:3]
        ey = hetg.elements[i].geometry['e_y'][:3]
        ez = hetg.elements[i].geometry['e_z'][:3]
        x = np.array([hetg.hess['xu'][i], hetg.hess['yu'][i], hetg.hess['zu'][i]])
        y = np.array([hetg.hess['xuxf'][i], hetg.hess['yuxf'][i], hetg.hess['zuxf'][i]])
        z = np.array([hetg.hess['xuyf'][i], hetg.hess['yuyf'][i], hetg.hess['zuyf'][i]])
        # all a normalized
        for t in [ex, ey, ez, x, y, z]:
            assert np.allclose(np.linalg.norm(t), 1., atol=1e-4)
        # angle between them is small
        assert np.abs(np.dot(x, ex)) > 0.99
        assert np.abs(np.dot(y, ey)) > 0.99
        assert np.abs(np.dot(z, ez)) > 0.99


def test_groove_dir():
    '''Regression test: Groove direction was not set correctly from input table.'''
    hetg = HETG()
    for i in range(len(hetg.elements)):
        a = hetg.elements[i].geometry['e_groove'][:3]
        b = np.array([hetg.hess['xul'][i], hetg.hess['yul'][i], hetg.hess['zul'][i]])
        c = hetg.elements[i].geometry['e_perp_groove'][:3]
        d = np.array([hetg.hess['xud'][i], hetg.hess['yud'][i], hetg.hess['zud'][i]])
        # all a normalized
        for x in [a, b, c, d]:
            assert np.allclose(np.linalg.norm(x), 1., atol=1e-4)
        # angle between them is small
        assert np.abs(np.dot(a, b)) > 0.99
        assert np.abs(np.dot(c, d)) > 0.99
