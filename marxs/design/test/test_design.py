import numpy as np

from ..rowland import RowlandTorus

def parametrictorus(R, r, theta, phi):
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z

def test_torus():
    '''Test the torus equation for a set of points.

    Importantly, we use a parametric description of the torus here, which is
    different from the actual implementation.
    '''
    R = 2.
    r = 1.

    angle = np.arange(0, 2 * np.pi, 0.1)
    phi, theta = np.meshgrid(angle, angle)
    phi = phi.flatten()
    theta = theta.flatten()
    x, y, z = parametrictorus(R, r, theta, phi)
    mytorus = RowlandTorus(R=R, r=r)

    assert np.allclose(mytorus.quartic(x, y, z), 0.)

def test_torus_normal():
    R = 2.
    r = 1.

    angle = np.arange(0, 2 * np.pi, 0.1)
    phi, theta = np.meshgrid(angle, angle)
    phi = phi.flatten()
    theta = theta.flatten()
    x, y, z = parametrictorus(R, r, theta, phi)
    xt1, yt1, zt1 = parametrictorus(R, r, theta + 0.001, phi)
    xp1, yp1, zp1 = parametrictorus(R, r, theta, phi + 0.001)
    xt0, yt0, zt0 = parametrictorus(R, r, theta - 0.001, phi)
    xp0, yp0, zp0 = parametrictorus(R, r, theta, phi + 0.001)

    mytorus = RowlandTorus(R=R, r=r)
    vec_normal = np.vstack(mytorus.normal(x, y, z))

    vec_delta_theta = np.vstack([xt1 - xt0, yt1 - yt0, zt1 - zt0])
    vec_delta_phi = np.vstack([xp1 - xp0, yp1 - yp0, zp1 - zp0])

    assert np.allclose(np.einsum('ij,ij->j', vec_normal, vec_delta_theta), 0.)
    assert np.allclose(np.einsum('ij,ij->j', vec_normal, vec_delta_phi), 0.)
