import numpy as np

from .pluecker import e2h
from .utils import norm_vector

def polarization_vectors(dir_array, angles):
    '''Takes angle polarizations and converts them to vectors in the direction of polarization.

    Follows convention: Vector perpendicular to photon direction and closest to +y axis is
    angle 0 for polarization direction, unless photon direction is parallel to the y axis, in
    which case the vector closest to the +x axis is angle 0.

    Parameters
    ----------
    dir_array : nx4 np.array
        each row is the homogeneous coordinates for a photon's direction vector
    angles : np.array
        1D array with the polarization angles
    '''
    n = len(angles)
    polarization = np.zeros((n, 4))
    x = np.array([1., 0., 0.])
    y = np.array([0., 1., 0.])

    #	NOTE: The commented code works and is more readable, but the current code is faster.
    #	for i in range(0, n):
    #		r = h2e(dir_array[i])
    #		r /= np.linalg.norm(r)
    #		if not (np.isclose(r[0], 0.) and np.isclose(r[2], 0.)):
    #			# polarization relative to positive y at 0
    #			v_1 = y - (r * np.dot(r, y))
    #			v_1 /= np.linalg.norm(v_1)
    #		else:
    #			# polarization relative to positive x at 0
    #			v_1 = x - (r * np.dot(r, x))
    #			v_1 /= np.linalg.norm(v_1)
    #
    #		# right hand coordinate system is v_1, v_2, r (photon direction)
    #		v_2 = np.cross(r, v_1)
    #		polarization[i, 0:3] = v_1 * np.cos(angles[i]) + v_2 * np.sin(angles[i])
    #		polarization[i, 3] = 0

    r = dir_array.copy()[:,0:3]
    r /= np.linalg.norm(r, axis=1)[:, np.newaxis]
    pol_convention_x = np.isclose(r[:,0], 0.) & np.isclose(r[:,2], 0.)
    # polarization relative to positive y or x at 0
    v_1 = ~pol_convention_x[:, np.newaxis] * (y - r * np.dot(r, y)[:, np.newaxis])
    v_1 += pol_convention_x[:, np.newaxis] * (x - r * np.dot(r, x)[:, np.newaxis])
    v_1 /= np.linalg.norm(v_1, axis=1)[:, np.newaxis]

    # right hand coordinate system is v_1, v_2, r (photon direction)
    v_2 = np.cross(r, v_1)
    polarization[:, 0:3] = v_1 * np.cos(angles)[:, np.newaxis] + v_2 * np.sin(angles)[:, np.newaxis]

    return polarization

def Q_reflection(delta_dir):
    '''Reflection of a polarization vector on a non-polarizing surface.

    This can also be used for other elements that change the direction of the
    photon without adding any more polarization and where both sides
    propagate in the same medium.
    See `Yun (2011) <http://hdl.handle.net/10150/202979>`_, eqn 4.3.13 for details.

    Parameters
    ----------
    delta_dir : np.array of shape (n, 4)
        Array of photon direction coordinates in homogeneous coordinates:
        ``delta_dir = photon['dir_old'] - photons['dir_new']``.
        Note that this vector is **not** normalized.

    Returns
    -------
    q : np.array of shape (n, 4, 4)
        Array of parallel transport ray tracing matrices.
    '''
    if delta_dir.shape != 2:
        raise ValueError('delta_dir must have dimension (n, 4).')
    m = delta_dir[..., None, :] * delta_dir[..., :, None]
    return np.eye(4) - 2 / (np.linalg.norm(delta_dir, axis=1)**2)[:, None, None] * m

def paralleltransport_matrix(dir1, dir2, jones=np.eye(2), replace_nans=True):
    '''Calculate parallel transport ray tracing matrix.

    Parallel transport for a vector implies that the component s
    (perpendicular, from German *senkrecht*) to the planes spanned by
    ``dir1`` and ``dir2`` stays the same. If ``dir1`` is parallel to ``dir2``
    this plane is not well defined and the resulting matrix elements will
    be set to ``np.nan``, unless ``replace_nans`` is set.

    Note that the ray matrix returned works on an eukledian 3d vector, not a
    homogeneous vector. (Polarization is a vector, thus the forth element of the
    homogeneous vector is always 0 and returning (4,4) matrices is just a waste
    of space.)

    Parameters
    ----------
    dir1, dir2 : np.array of shape (n, 3)
        Direction before and after the interaction.
    jones : np.array of shape (2,2)
        Jones matrix in the local s,p system of the optical element.
    replace_nans : bool
        If ``True`` return an identity matrix for those rays with
        ``dir1=dir2``. In those cases, the local coordinate system is not well
        defines and thus no Jones matrix can be applied. In MARXS ``dir1=dir2``
        often happens if some photons in a list miss the optical element in
        question - these photons just pass through.

    Returns
    -------
    p_mat : np.array of shape(n, 3, 3)
    '''
    dir1 = norm_vector(dir1)
    dir2 = norm_vector(dir2)
    jones_3 = np.eye(3)
    jones_3[:2, :2] = jones
    s = np.cross(dir1, dir2)
    s = s / np.linalg.norm(s, axis=1)[:, None]
    p_in = np.cross(dir1, s)
    p_out = np.cross(dir2, s)

    Oininv = np.array([s, p_in, dir1]).swapaxes(1, 0)
    Oout = np.array([s, p_out, dir2]).swapaxes(1, 2).T
    temp = np.einsum('...ij,kjl->kil', jones_3, Oininv)
    pmat = np.einsum('ijk,ikl->ijl', Oout, temp)

    if replace_nans:
        ind = np.isnan(s[:, 0])
        pmat[ind, :, :] = np.eye(3)[None, :, :]
    return pmat

def parallel_transport(dir_old, dir_new, pol_old, **kwargs):
    '''Parallel transport of the polarization vector with no polarization happening.

    Parameters
    ----------
    dir_old, dir_new : np.array of shape (n, 4)
        Old and new photon direction in homogeneous coordinates.
    pol_old : np.array of shape (n, 4)
        Old polarization vector in homogeneous coordinates.
    kwargs : dict
        All other arguments are passed on to `~marxs.math.polarization.paralleltransport_matrix`.
    Returns
    -------
    pol : np.array of shape (m, 4)
        Parallel transported vectors.
    '''
    pmat = paralleltransport_matrix(dir_old[:, :3], dir_new[:, :3])
    out = np.einsum('ijk,ik->ij', pmat, pol_old[:, :3])
    return e2h(out, 0)
