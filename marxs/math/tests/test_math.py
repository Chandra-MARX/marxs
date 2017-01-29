# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
import pytest

from ..pluecker import *

pos1d100 = np.array([1.,0,0,1])
pos1d100_a = np.array([-3., 0, 0, -3])
pos1d200 = np.array([2.,0,0,1])
pos1d333 = np.array([3. , 3, 3, 1])
pos2d000 = np.array([[0.,0,0,1],[0,0,0,2]])
pos2d100 = np.array([[1.,0,0,1],[2,0,0,2]])


# This might not actually need this many tests, but now I have them.
@pytest.mark.parametrize("in1,in2,expected", zip([pos1d100, pos1d100_a, pos2d000, pos2d100], [pos1d200, pos2d100, pos1d200, pos2d000], [1, np.zeros(2), 2*np.ones(2), np.ones(1)]))
def test_distance_point_point(in1, in2, expected):
    '''Test euklidean distance between point in homogeneous coordinates.'''
    assert np.all(distance_point_point(in1, in2) == expected)


def test_pointdir2plane():
    point = np.random.random(4) - 0.5
    # cannot be 0, and too small number might cause numeric inaccuracy
    point[3] = max(point[3], 0.1)
    direction = np.random.random(4) - 0.5
    direction[3] = 0
    p = point_dir2plane(point, direction)
    assert np.allclose(np.dot(p, point), 0)
    assert np.allclose(np.cross(p[:3], direction[:3]), 0)


def test_dir_point2line():
    '''Construct Plucker coordinates from direction and point

    This example is taken from
    Pluecker Coordinate Tutorial, by Ken Shoemake
    '''
    p = np.array([2., 3, 7])
    q = np.array([2., 1, 0])
    line = np.array([0., 2, 7, -7, 14, -4])
    assert np.all(dir_point2line(p - q, p) == line)
    # Same calculation, but 2 dim inputs
    p = np.tile(p, (2, 1))
    q = np.tile(q, (2, 1))
    lines = np.tile(line, (2, 1))
    assert np.all(dir_point2line(p - q, p) == lines)


def test_intersect_line_plane():
    q = np.array([2., 1, 0])
    line = np.array([0., 2, 7, -7, 14, -4])
    assert np.all(h2e(intersect_line_plane(line, np.array([0., 0., 1., 0.]))) == q)
    # Similar calculation but for several input lines.
    # First makes the lines
    p = np.array([[3., 5, 2], [-4, 5., 2.]])
    # Very small chance that I generate a line that's in the plane.
    # If that ever comes up in tests, hardcode something better.
    q = np.random.random((2, 3))
    lines = e_pointpoint2line(p, q)
    plane = np.array([0, -1., 0, 5])
    assert np.allclose(h2e(intersect_line_plane(lines, plane)), p)


def test_intersect_line_plane_parallel():
    '''Check results for a line that is parallel to or even in the plane.'''
    # xy plane at z = +20
    plane = np.array([0, 0, 1, -20.])
    line_in_plane = dir_point2line(np.array([1, 5, 0]), np.array([5,-500,20]))
    assert np.all(intersect_line_plane(line_in_plane, plane) == 0)

    line_parallel_plane = dir_point2line(np.array([-.1, 0., 0.]), np.ones(3))
    intersect = intersect_line_plane(line_parallel_plane, plane)
    assert intersect[3] == 0
    assert np.any(intersect != 0)
