import numpy as np
import pytest

from ..rotations import ex2vec_fix

def is_orthogonal(a):
    '''Return True is a matrix is orthonormal'''
    try:
        if (a.ndim == 2) and (a.shape[0] == a.shape[1]):
            return np.allclose(np.dot(a, a.T), np.eye(a.shape[0]))
        else:
            return False
    # e.g. `a` is not a matirx
    except:
        return False

def is_specialorthogonal(a):
    return is_orthogonal(a) and np.allclose(np.linalg.det(a), 1)

def test_ex2vec_fix():
    '''Valid inputs. All results have to be rotation matrices.'''
    assert np.all(ex2vec_fix(np.array([1., 0, 0]), np.array([0., 1, 0])) == np.eye(3))

    for i in [1., 1.2, 3.4, 5.6]:
        t = i + np.arange(3)
        rot = ex2vec_fix(t, np.array([0., 1. ,0.]))
        assert is_specialorthogonal(rot)

def test_ex2vec_fix_invalid_input():
    '''Rotation is not well defined, if both input vectors are parallel.'''
    with pytest.raises(ValueError) as e:
        v = np.array([1.2, 3.4, 5.6])
        t = ex2vec_fix(v, 3 * v)
    assert 'are parallel' in str(e.value)

