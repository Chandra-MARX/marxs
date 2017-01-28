# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from transforms3d.utils import normalized_vector

def ex2vec_fix(e1, efix):
    '''Rotate x-axis to e1, use efix to break rotation ambiguity.

    This function calcalates the rotation matrix that rotates the x-axis to
    ``e1``, i.e. it rotates the normal of the y,z-plane to a new plane where
    ``e1`` is the normal. This still leaves the rotation angle of the plane
    free. This function breaks the degeneracy by keeping a vector that is
    coplanar with the normal and fix coplanar with fix and the new normal.

    The purpose of this function is best explained by an example:
    Imagine a plane in the y, z plane. We want to rotate the plane, such that ``e1``
    will be a new normal of the plane. However, since the orientation of the plane
    is not fixed, there is an infinite number of ways to do this.
    This ambiguity can be broken if we additionally require that the rotated
    :math:`\hat e_y` will be coplanar with ``fix``.

    Inspired by the algorithm described in:
    https://www.fastgraph.com/makegames/3drotation/

    Parameters
    ----------
    e1 : np.array of shape (3, )
        new normal of plane
    efix : np.array of shape (3, )
        Vector to break rotational ambiguity.

    Returns
    -------
    rot : np.array of shape (3, 3)
        Rotation matrix
    '''
    e1 = normalized_vector(e1)
    efix = normalized_vector(efix)
    if np.allclose(e1, efix) or np.allclose(e1, -efix):
        raise ValueError('Input vectors are parallel - Rotation matrix is ambiguous.')
    rot = np.empty((3, 3))
    rot[:, 0] = e1
    rot[:, 1] = normalized_vector(efix - np.dot(efix, e1) * e1)
    rot[:, 2] = np.cross(rot[:, 0], rot[:, 1])
    return rot


def axangle2mat(axes, angles, is_normalized=False):
    ''' Rotation matrix for rotation angle `angle` around `axis`

    This is a vectorized version of the routine of the same name in
    ``transforms3d``.

    Parameters
    ----------
    axes : np.array of shape (N, 3)
       vector specifying axis for rotation.
    angle : np.array
       angle of rotation in radians.
    is_normalized : bool, optional
       True if `axis` is already normalized (has norm of 1).  Default False.

    Returns
    -------
    mat : array shape (N, 3,3)
       rotation matrices for specified rotation

    Notes
    -----
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    '''
    if len(angles) != axes.shape[0]:
        raise ValueError('There must be one angle for each axes vector.')

    if not is_normalized:
        axes = axes / np.linalg.norm(axes, axis=1)[:, None]
    c = np.cos(angles); s = np.sin(angles); C = 1-c
    x = axes[:, 0]; y = axes[:, 1]; z = axes[:, 2]
    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC
    return np.array([
            [ x*xC+c,   xyC-zs,   zxC+ys ],
            [ xyC+zs,   y*yC+c,   yzC-xs ],
            [ zxC-ys,   yzC+xs,   z*zC+c ]]).swapaxes(0,2).swapaxes(1,2)
