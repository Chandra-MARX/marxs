# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from transforms3d import affines, euler

__all__ = ['generate_facet_uncertainty']


def generate_facet_uncertainty(n, xyz, angle_xyz,
                               trans_offset=0, rot_offset=0):
    '''Generate 4d matrices that represent facet misalignment.

    Positional and rotational uncertainties are input to this function. It then
    draws randomnly from Gaussians centered on 0 (the correct position) for the displacement
    and rotation, where the :math:`\sigma` of the Gaussian is given by the numbers in the input.
    The linear displacements and angles are expressed as (4,4) matrixes suitable for use with
    homogeneous coordinates.

    Parameters
    ----------
    n : int
        Number of 4d matrixes to be calculated
    xyz : tuple of 3 floats
        accuracy of grating positioning in x, y, z (in mm) - Gaussian sigma, not FWHM!
    angle_xyz : tuple of 3 floats
        accuracy of grating positioning. Rotation around x, y, z (in rad) - Gaussian sigma, not FWHM!
    trans_offset : array_like
        will be passed to `numpy.random.normal` as the ``loc`` parameter when drawing translantions.
    rot_offset : array_like
        will be passed to `numpy.random.normal` as the ``loc`` parameter when drawing rotations.

    Returns
    -------
    pos_uncert : list of n (4,4) np.arrays
        Random realizations of the uncertainty
    '''
    translation = np.random.normal(size=(n, 3), loc=trans_offset, scale=xyz)
    rotation = np.random.normal(size=(n, 3), loc=rot_offset, scale=angle_xyz)
    return [affines.compose(t, euler.euler2mat(a[0], a[1], a[2], 'sxyz'), np.ones(3))
            for t, a in zip(translation, rotation)]