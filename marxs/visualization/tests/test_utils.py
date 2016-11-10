import numpy as np
import pytest

from ..utils import plane_with_hole, get_color, color_tuple_to_hex, format_saved_positions

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

def test_color_roundtrip():
    '''Test that the different color convertes are consistent.'''
    colortup = get_color({'color': 'white'})
    assert colortup == (1.0, 1.0, 1.0)
    hexstr = color_tuple_to_hex(colortup)
    assert hexstr == '0xffffff'
    assert get_color({'color': '#ffffff'}) == (1.0, 1.0, 1.0)
    # repeat with a second color that has different values for rgb
    colortup = get_color({'color': '#ff013a'})
    # Note that matplotlib expects hex strings with '#' while python uses '0x'
    assert color_tuple_to_hex(colortup) == '0xff013a'

    # Int input
    assert color_tuple_to_hex((255, 255, 255)) == '0xffffff'

def test_color_hex_pad():
    '''Regression test: We want to color hexstring with leading zeroths'''
    assert color_tuple_to_hex((0., 1., .5)) == '0x00ff7f'

def test_color_to_hex_bad_input():
    with pytest.raises(ValueError) as e:
        out = color_tuple_to_hex('white')
    assert 'Input tuple must be all' in str(e.value)

    with pytest.raises(ValueError) as e:
        out = color_tuple_to_hex((-1, 0, 234))
    assert 'Int values in color tuple' in str(e.value)

    with pytest.raises(ValueError) as e:
        out = color_tuple_to_hex((1., 3., 0.3))
    assert 'Float values in color tuple' in str(e.value)

def test_format_saved_positions():
    '''Reformat saved positions and drop nearly identical values.'''
    class A(object):
        pass

    pos0 = np.arange(20).reshape(5,4)
    pos1 = pos0 + 1
    pos2 = pos1 + 1e-4
    pos = A()
    pos.data = [pos0, pos1, pos2]

    d = format_saved_positions(pos)
    assert d.shape == (5, 2, 3)
    assert np.allclose(d[0, 0, :], [0, 1./3, 2./3])

def test_empty_format_saved_positions():
    '''If the input contains no data, an error should be raised.'''
    class A(object):
        data = []

    a = A()
    with pytest.raises(ValueError) as e:
        d = format_saved_positions(a)

    assert 'contains no data' in str(e.value)
