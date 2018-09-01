# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np


def e2h(e, w):
    '''Convert Euclidean coordinates to homogeneous coordinates

    Parameters
    ----------
    e : np.array
        Input Euclidean coordinates. This can be multidimensional, but the last
        dimension must be of size 3.
    w : float
        ``0`` for directions (points at infinity)
        ``1`` for positions (points in Euclidean space)

    Returns
    -------
    h : np.array
        Homogeneous coordinates. Same shape as ``e`` except that the last
        dimension is now has 4 elements.
    '''
    if not ((w == 0) or (w == 1)):
        raise ValueError('w must be 0 or 1.')
    shape = list(e.shape)
    shape[-1] += 1
    h = np.empty(shape)
    h[..., :3] = e
    h[..., 3] = w
    return h


def h2e(h):
    '''Convert homogeneous coordinates to Euclidean coordinates

    This functions works for points at infinity and for points in real
    euclidean space, but it expects that each time it is called ``h`` contains
    only the one or the other but not a mixture of both.

    Parameters
    ----------
    h : np.array
        Input homogeneous coordinates. This can be multidimensional, but the
        last dimension must be of size 4.

    Returns
    -------
    e : np.array
        Euclidean coordinates. Same shape as ``e`` except that the last
        last dimension is now has 3 elements.
    '''
    if np.all(h[..., 3] == 0) or np.allclose(h[..., 3], 1):
        return h[..., :3]
    elif np.all(h[..., 3] != 0):
        return (h[..., :3] / h[..., 3][..., None])
    else:
        raise ValueError('Input array must be either all euklidean points or all points at infinity.')


def distance_point_point(h_p1, h_p2):
    '''Calculate the Eukledian distance between 2 points

    Parameters
    ----------
    h_p1, h_p2 : np.ndarray of shape (4) or (N, 4)
        homogeneous coordiantes of input points

    Returns
    -------
    d : np.array of shape (1) or (N)
        Eukledian distance between those points
    '''
    if max(h_p1.ndim, h_p2.ndim) == 1:
        axis = 0
    elif max(h_p1.ndim, h_p2.ndim) == 2:
        axis = 1
    else:
        raise ValueError('This function expects 1d or 2d input.')
    return np.linalg.norm(h2e(h_p1) - h2e(h_p2), axis=axis)


def translation2aff(vec):
    '''Transform 3-d translation vector to affine 4*4 matrix.

    Parameters
    ----------
    vec : (3, ) array
        x, y, z translation vector

    Returns
    -------
    m : (4, 4) array
        affine transformation matrix
    '''
    if len(vec) != 3:
        raise ValueError('3d translation vector expected.')
    m = np.eye(4)
    m[:3, 3] = vec
    return m


def zoom2aff(zoom):
    '''Transform zoom to affine 4*4 matrix.

    Parameters
    ----------
    vec : float or (3, ) array
        zoom factor for x, y, z dimension (the same zoom is applied to
        every dimension is the input is a scalar).

    Returns
    -------
    m : (4, 4) array
        affine transformation matrix
    '''
    z = np.eye(4)
    if np.isscalar(zoom):
        z[:3,:3] = zoom * z[:3, :3]
    elif len(zoom) == 3:
        z[:3, :3] = np.diag(zoom)
    else:
        raise ValueError('Zoom must be either scalar or 3 element vector.')
    return z


def mat2aff(mat):
    '''Transform 3*3 matrix (e.g. rotation) to affine 4*4 matrix.

    Parameters
    ----------
    mat : (3, 3) array
        input matrix

    Returns
    -------
    m : (4, 4) array
        affine transformation matrix
    '''
    m = np.eye(4)
    m[:3, :3] = mat
    return m

def norm_vector(vec):
    '''Normalize euklidean vectors.

    Parameters
    ----------
    vec : np.array
        Input vectors of shape (n, 3)

    Returns
    -------
    vec : np.array
        Normalized vectors
    '''
    length2 = np.sum(vec * vec, axis=-1)
    return vec / np.sqrt(length2)[:, None]

def anglediff(phi):
    '''Angle range covered by phi, accounting for 2 pi properly

    Parameters
    ----------
    phi : list of two float
        Two angles in radian
    '''
    anglediff = phi[1] - phi[0]
    if (anglediff < 0.) or (anglediff > (2. * np.pi)):
        # If anglediff == 2 pi exactly, presumably the user want to cover the full circle.
        anglediff = anglediff % (2. * np.pi)
    return anglediff

def angle_between(angle, border1, border2):
    '''Test if an angle is between two others

    Since angles are cyclic, a simple numerical comparison is not
    sufficient, e.g. 355 deg is in the interval [350 deg, 10 deg] even though
    numerically 355 is not less then 10.

    Parameters
    ----------
    angle : float or np.array
        Angle array (in rad)
    border1 : float
        Lower border of interval (inclusive) in radian
    border2 : float
        Higher border of interval (inclusive) in radian

    Returns
    -------
    comparison : bool of same shape as ``angle``
        Result of the comparison

    Examples
    --------

    >>> from marxs.math.utils import angle_between
    >>> angle_between(-0.1, -0.2, 6)
    True
    '''
    twopi = 2 * np.pi
    b1 = (twopi + (border1 % twopi)) % twopi
    b2 = (twopi + (border2 % twopi)) % twopi
    ang = (twopi + (angle % twopi)) % twopi
    if b1 < b2:
        return (b1 <= ang) & (ang <= b2)
    else:
        return (b1 <= ang) | (ang <= b2)
