import numpy as np
from scipy.stats import kstest
import transforms3d
import pytest

from ..rowland import RowlandTorus, GratingArrayStructure, find_radius_of_photon_shell, design_tilted_torus
from ...optics.base import OpticalElement
from ...source import ConstantPointSource, FixedPointing
from ...optics import MarxMirror, uniform_efficiency_factory, FlatGrating

def parametrictorus(R, r, theta, phi):
    '''Just another way to specify a torus with z-axis as symmetry'''
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return np.array([x, y, z]).T

def test_radius_of_photon_shell():
    mysource = ConstantPointSource((30., 30.), flux=1., energy=1.)
    photons = mysource.generate_photons(1000)
    mypointing = FixedPointing(coords=(30, 30.))
    photons = mypointing.process_photons(photons)
    marxm = MarxMirror('./marxs/optics/hrma.par', position=np.array([0., 0, 0]))
    photons = marxm.process_photons(photons)
    r1, r2 = find_radius_of_photon_shell(photons, 3, 9e4)
    assert abs(r1 - 5380.) < 10.
    assert abs(r2 - 5495.) < 10.

def test_design_tilted_torus():
    '''Test the trivial case with analytic answers and consistency for other angles'''
    R, r, pos4d = design_tilted_torus(10, 0., 0.)
    assert R == 5
    assert r == 5
    assert np.all(pos4d == np.eye(4))
    # This is a solution that looks good by hand, but could still be slightly wrong...
    R, r, pos4d = design_tilted_torus(10, np.deg2rad(3), np.deg2rad(6))
    assert r == 5.0068617299896054
    assert R == 4.9794336175561327
    assert pos4d[0,3] == 0.027390523158633273

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
    xyz = parametrictorus(R, r, theta, phi)
    mytorus = RowlandTorus(R=R, r=r)

    assert np.allclose(mytorus.quartic(xyz), 0.)
    # This torus has pos4d=eye(4), so the same results should come out with this shortcut
    assert np.allclose(mytorus.quartic(xyz, transform=False), 0.)

def test_torus_solve_quartic():
    '''solve_quartic helps to find point on the torus if two coordinates are fixed.'''
    xyz = parametrictorus(2., 1., [.34, 1.23], [.4, 4.5])
    mytorus = RowlandTorus(2., 1.)
    for i in [0, 1]:
        for j in range(3):
            kwargs = {'x': xyz[i, 0], 'y': xyz[i, 1], 'z': xyz[i, 2],
                      'interval' : np.array([-0.1, 0.1]) + xyz[i, j]}
            kwargs['xyz'[j]] = None
            assert np.allclose(xyz[i, j], mytorus.solve_quartic(**kwargs))

    # Nothing to solve for
    with pytest.raises(ValueError) as e:
        out = mytorus.solve_quartic(x=1, y=1, z=1)
    assert 'Exactly one of the input numbers' in str(e.value)

    # Too many parameters to solve for
    with pytest.raises(ValueError) as e:
        out = mytorus.solve_quartic(x=1, y=None, z=None)
    assert 'Exactly one of the input numbers' in str(e.value)

def test_rotated_torus():
    '''Test the torus equation for a set of points.

    Importantly, we use a parametric description of the torus here, which is
    different from the actual implementation.
    '''
    angle = np.arange(0, 2 * np.pi, 0.1)
    phi, theta = np.meshgrid(angle, angle)
    phi = phi.flatten()
    theta = theta.flatten()
    xyz = parametrictorus(2., 1., theta, phi)
    # rotate -90 deg around y axis:
    #   x -> z
    #   z -> -x
    rotatedxyz = np.zeros_like(xyz)
    rotatedxyz[:, 0] = xyz[:, 2]
    rotatedxyz[:, 1] = xyz[:, 1]
    rotatedxyz[:, 2] = -xyz[:, 0]
    mytorus = RowlandTorus(R=2., r=1., orientation=transforms3d.axangles.axangle2mat([0,1,0], -np.pi/2))
    assert np.allclose(mytorus.quartic(rotatedxyz), 0.)


def test_torus_normal():
    '''Compare the analytic normal with a numeric result.

    The test computes a vector along the torus for small d_theta
    and d_phi. Both of those should be perpendicular to the normal.
    '''
    R = 2.
    r = 1.

    angle = np.arange(0, 2 * np.pi, 0.1)
    phi, theta = np.meshgrid(angle, angle)
    phi = phi.flatten()
    theta = theta.flatten()
    xyz = parametrictorus(R, r, theta, phi)
    t1 = parametrictorus(R, r, theta + 0.001, phi)
    p1 = parametrictorus(R, r, theta, phi + 0.001)
    t0 = parametrictorus(R, r, theta - 0.001, phi)
    p0 = parametrictorus(R, r, theta, phi + 0.001)

    mytorus = RowlandTorus(R=R, r=r)
    vec_normal = mytorus.normal(xyz)

    vec_delta_theta = t1 - t0
    vec_delta_phi = p1 - p0

    assert np.allclose(np.einsum('ij,ij->i', vec_normal, vec_delta_theta), 0.)
    assert np.allclose(np.einsum('ij,ij->i', vec_normal, vec_delta_phi), 0.)

def test_GratingArrayStructure():
    '''Check a GAS for consistency

    The expected numbers for this one are hand-checked and plotted and thus
    can be taken as known-good to prevent regressions later.
    '''
    myrowland = RowlandTorus(10000., 10000.)
    class mock_facet(OpticalElement):
        pass
    gas = GratingArrayStructure(myrowland, 30., [10000., 20000.], [300., 600.], phi=[-0.2*np.pi, 0.2*np.pi], facetclass=mock_facet)
    assert gas.max_facets_on_arc(300.) == 12
    angles = gas.distribute_facets_on_arc(315.) % (2. * np.pi)
    # This is a wrap-around case. Hard to test in general, but here I know the numbers
    assert np.alltrue((angles < 0.2 * np.pi) | (angles > 1.8 * np.pi))
    assert gas.max_facets_on_radius() == 10
    assert np.all(gas.distribute_facets_on_radius() == np.arange(315., 600., 30.))
    assert len(gas.facet_pos) == 177
    center = gas.calc_ideal_center()
    assert center[1] == 0
    assert center[2] == (300. + 600.) / 2.
    assert 2e4 - center[0] < 20.  # R_rowland >> r_gas  -> center close to R_rowland
    # fig, axes = plt.subplots(2,2)
    # for elem in [[axes[0, 0], 0, 1], [axes[0, 1], 0, 2], [axes[1, 0], 1, 2]]:
    #     for f in gas.facet_pos:
    #         elem[0].plot(f[elem[1],3], f[elem[2],3], 's')
    # plt.show()  # or move mouse in window or something.

    # There is a test missing here that the rotational part works correctly.
    # I just cannot think of a good way to check that right now.

    # Check that initially all uncertainties are 0
    for i in range(len(gas.facet_pos)):
        assert np.allclose(gas.facet_pos[i], gas.facets[i].pos4d)

def test_GratingArrayStructure_2pi():
    '''test that delta_phi = 2 pi means "full circle" and not 0
    '''
    myrowland = RowlandTorus(10000., 10000.)
    class mock_facet(OpticalElement):
        pass
    gas = GratingArrayStructure(myrowland, 30., [10000., 20000.], [300., 600.], phi=[0, 2*np.pi], facetclass=mock_facet)
    assert gas.max_facets_on_arc(300.) > 10
    n = len(gas.facet_pos)
    yz = np.empty((n, 2))
    for i, p in enumerate(gas.facet_pos):
        yz[i, :] = p[1:3, 3]
    phi = np.arctan2(yz[:, 1], yz[:, 0])
    ks, pvalue = kstest((phi + np.pi) / (2 * np.pi), 'uniform')
    assert pvalue > 0.3  # It's not exactly uniform because of finite size of facets.

def test_facet_rotation_via_facetargs():
    '''The numbers for the blaze are not realistic.'''
    gratingeff = uniform_efficiency_factory()
    mytorus = RowlandTorus(9e4/2, 9e4/2)
    mygas = GratingArrayStructure(mytorus, d_facet=60., x_range=[5e4,1e5], radius=[5380., 5500.], facetclass=FlatGrating, facetargs={'zoom': 30, 'd':0.0002, 'order_selector': gratingeff})
    blaze = transforms3d.axangles.axangle2mat(np.array([0,1,0]), np.deg2rad(15.))
    mygascat = GratingArrayStructure(mytorus, d_facet=60., x_range=[5e4,1e5], radius=[5380., 5500.], facetclass=FlatGrating, facetargs={'zoom': 30, 'orientation': blaze, 'd':0.0002, 'order_selector': gratingeff})
    assert np.allclose(np.rad2deg(np.arccos(np.dot(mygas.facets[0].geometry['e_x'][:3], mygascat.facets[0].geometry['e_x'][:3]))), 15.)

def test_persistent_facetargs():
    '''Make sure that facet_args is still intact after generating facets.

    This is important to allow tweaks of a single parameter and then to regerante the facets.
    '''
    gratingeff = uniform_efficiency_factory()
    mytorus = RowlandTorus(9e4/2, 9e4/2)
    facet_args = {'zoom': 30, 'd':0.0002, 'order_selector': gratingeff}
    mygas = GratingArrayStructure(mytorus, d_facet=60., x_range=[5e4,1e5], radius=[5380., 5500.], facetclass=FlatGrating, facetargs=facet_args)
    assert mygas.facet_args == facet_args
