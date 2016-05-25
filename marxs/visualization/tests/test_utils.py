import numpy as np

from ..utils import plane_with_hole

def test_hole_round():
    '''Ensure that plane around the inner hole closes properly at the last point.'''
    outer = np.array([[-1, -1, 1, 1], [-1, 1, 1, -1], [0,0,0,0]]).T

    # Try out a rectangle and a circle - the two cases we are most likely to need.
    for n in [3, 4, 90, 360]:
        inner  = np.zeros((n, 3))
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        inner[:, 0] = np.sin(phi)
        inner[:, 1] = np.cos(phi)
        xyz, triangles = plane_with_hole(outer, inner)
        assert triangles[0][1] == triangles[-1][2]
        tri = np.array(triangles)
        # Technically, the order of the points in each triangle
        # is irrelevant, but the current implementation does it this way
        # and that's an easy way to check.
        # Check first point is always on outer rim
        assert set(tri[:, 0]) == set([0,1,2,3])
        # Check all points turn up in the middle
        assert set(tri[:, 1]) == set(np.arange(4 + n))
        # Check last point is always on inner rim
        assert set(tri[:, 2]) == set(np.arange(4, 4 + n))
