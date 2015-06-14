import numpy as np

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
