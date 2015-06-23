import numpy as np
from transforms3d.utils import normalized_vector

def ex2vec_fix(e1, efix):
    '''Rotate x-axis to e1, keeping a vector that is coplanar with fix coplanar.

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

