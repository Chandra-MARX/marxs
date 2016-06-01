import numpy as np
from scipy.stats import kstest
import transforms3d
import pytest

from ..rowland import (RowlandTorus, GratingArrayStructure, LinearCCDArray,
                       find_radius_of_photon_shell, design_tilted_torus,
                       ElementPlacementError)
from ...optics.base import FlatOpticalElement
from ...source import PointSource, FixedPointing
from ...optics import MarxMirror, uniform_efficiency_factory, FlatGrating
from ...math.pluecker import h2e

class mock_facet(FlatOpticalElement):
    '''Lightweight class with no functionality for tests.'''
    pass

def test_radius_of_photon_shell():
    mysource = PointSource((30., 30.), flux=1., energy=1.)
    photons = mysource.generate_photons(1000)
    mypointing = FixedPointing(coords=(30, 30.))
    photons = mypointing.process_photons(photons)
    marxm = MarxMirror('./marxs/optics/hrma.par', position=np.array([0., 0, 0]))
    photons = marxm.process_photons(photons)
    r1, r2 = find_radius_of_photon_shell(photons, 0, 9e4)
    assert abs(r1 - 5380.) < 10.
    assert abs(r2 - 5495.) < 10.
    r1, r2 = find_radius_of_photon_shell(photons, 1, 9e3)
    assert abs(r1 - 433.) < 1.
    assert abs(r2 - 442.) < 1.
    r1, r2 = find_radius_of_photon_shell(photons, 1, 9e3, percentile=[49, 49.001])
    assert (r2 - r1) < 0.01

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
    # The z axis should be unchanged
    assert np.allclose(pos4d[[0,1],2], 0)

def test_design_tilted_torus_negative_angles():
    '''Tilt the torus the other way. Should be same geometry in general.'''
    R, r, pos4d = design_tilted_torus(10, np.deg2rad(3), np.deg2rad(6))
    Rn, rn, pos4dn = design_tilted_torus(10, np.deg2rad(-3), np.deg2rad(-6))
    assert R == Rn
    assert r == rn
    assert pos4d[0,3] == pos4dn[0, 3]
    assert pos4d[2, 3] == 0 # torus in xy plane
    assert pos4dn[2, 3] == 0
    assert pos4d[1, 3] == - pos4dn[1, 3]

def test_torus():
    '''Compare parametric equation of torus with non-parametric.
    '''
    R = 2.
    r = 1.

    angle = np.arange(0, 2 * np.pi, 0.1)
    phi, theta = np.meshgrid(angle, angle)
    phi = phi.flatten()
    theta = theta.flatten()
    mytorus = RowlandTorus(R=R, r=r)

    xyzw = mytorus.parametric(theta, phi)
    xyz = h2e(xyzw)

    assert np.allclose(mytorus.quartic(xyz), 0.)
    # This torus has pos4d=eye(4), so the same results should come out with this shortcut
    assert np.allclose(mytorus.quartic(xyz, transform=False), 0.)

def test_torus_solve_quartic():
    '''solve_quartic helps to find point on the torus if two coordinates are fixed.'''
    mytorus = RowlandTorus(2., 1.)
    xyzw = mytorus.parametric([.34, 1.23], [.4, 4.5])
    xyz = h2e(xyzw)

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
    def parametrictorus(R, r, theta, phi):
        '''Just another way to specify a torus with y-axis as symmetry'''
        x = (R + r * np.cos(theta)) * np.cos(phi)
        z = (R + r * np.cos(theta)) * np.sin(phi)
        y = r * np.sin(theta)
        return np.array([x, y, z]).T

    angle = np.arange(0, 2 * np.pi, 0.1)
    phi, theta = np.meshgrid(angle, angle)
    phi = phi.flatten()
    theta = theta.flatten()
    xyz = parametrictorus(2., 1., theta, phi)
    # rotate -90 deg around y axis:
    #   x -> z
    #   z -> -x
    rotatedxyz = np.zeros_like(xyz)
    rotatedxyz[:, 0] = -xyz[:, 2]
    rotatedxyz[:, 1] = xyz[:, 1]
    rotatedxyz[:, 2] = xyz[:, 0]
    mytorus = RowlandTorus(R=2., r=1., orientation=transforms3d.axangles.axangle2mat([0,1,0], -np.pi/2))

    assert np.allclose(mytorus.quartic(rotatedxyz), 0.)

    assert np.allclose(rotatedxyz, h2e(mytorus.parametric(theta, phi)))


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
    mytorus = RowlandTorus(R=R, r=r)

    xyz = mytorus.parametric(theta, phi)
    t1 = mytorus.parametric(theta + 0.001, phi)
    p1 = mytorus.parametric(theta, phi + 0.001)
    t0 = mytorus.parametric(theta - 0.001, phi)
    p0 = mytorus.parametric(theta, phi + 0.001)

    vec_normal = mytorus.normal(h2e(xyz))

    vec_delta_theta = h2e(t1) - h2e(t0)
    vec_delta_phi = h2e(p1) - h2e(p0)

    assert np.allclose(np.sqrt(np.sum(vec_normal * vec_normal, axis=1)), 1.)
    assert np.allclose(np.einsum('ij,ij->i', vec_normal, vec_delta_theta), 0.)
    assert np.allclose(np.einsum('ij,ij->i', vec_normal, vec_delta_phi), 0.)

def torus_cut_is_circle():
    '''Cutting the torus in a plane that includes the symmetry axis should give a cirlce.

    Check that that's true even for a tilted torus.
    '''
    R, r, pos4d = design_tilted_torus(10, np.deg2rad(-3), np.deg2rad(-6))
    torus = RowlandTorus(R, r, pos4d=pos4d)

    z = np.arange(0., torus.r *0.8, 5)
    x = np.zeros_like(z)
    xguess = torus.r + torus.R
    for i, iz in enumerate(z):
        x[i] = torus.solve_quartic(None, 0, iz, [0.99 * xguess, 1.1 * xguess])
        xguess = x[i]

    # This should be a circle of radius r around a center.
    # Where exactly is the center?
    # Assuming no y rotation:
    circle_center = np.dot(torus.pos4d[:3,:3], torus.R * np.array([1, 0,0])) + torus.pos4d[:3,3]
    circle_center_xz = circle_center[[0,2]]
    p = np.vstack([x,z])
    d = np.linalg.norm(p - circle_center_xz[:, np.newaxis], axis=0)
    assert np.allclose(d, torus.r)

def test_GratingArrayStructure():
    '''Check a GAS for consistency

    The expected numbers for this one are hand-checked and plotted and thus
    can be taken as known-good to prevent regressions later.
    '''
    myrowland = RowlandTorus(10000., 10000.)
    gas = GratingArrayStructure(myrowland, 30., [10000., 20000.], [300., 600.], phi=[-0.2*np.pi, 0.2*np.pi], elem_class=mock_facet)
    assert gas.max_elements_on_arc(300.) == 12
    angles = gas.distribute_elements_on_arc(315.) % (2. * np.pi)
    # This is a wrap-around case. Hard to test in general, but here I know the numbers
    assert np.alltrue((angles < 0.2 * np.pi) | (angles > 1.8 * np.pi))
    assert gas.max_elements_on_radius() == 10
    assert np.all(gas.distribute_elements_on_radius() == np.arange(315., 600., 30.))
    assert len(gas.elem_pos) == 177
    center = gas.calc_ideal_center()
    assert center[2] == 0
    assert center[1] == (300. + 600.) / 2.
    assert 2e4 - center[0] < 20.  # R_rowland >> r_gas  -> center close to R_rowland
    # fig, axes = plt.subplots(2,2)
    # for elem in [[axes[0, 0], 0, 1], [axes[0, 1], 0, 2], [axes[1, 0], 1, 2]]:
    #     for f in gas.elem_pos:
    #         elem[0].plot(f[elem[1],3], f[elem[2],3], 's')
    # plt.show()  # or move mouse in window or something.

    # There is a test missing here that the rotational part works correctly.
    # I just cannot think of a good way to check that right now.

    # Check that initially all uncertainties are 0
    for i in range(len(gas.elem_pos)):
        assert np.allclose(gas.elem_pos[i], gas.elements[i].pos4d)

def test_GratingArrayStructure_2pi():
    '''test that delta_phi = 2 pi means "full circle" and not 0
    '''
    myrowland = RowlandTorus(10000., 10000.)
    gas = GratingArrayStructure(myrowland, 30., [10000., 20000.], [300., 600.], phi=[0, 2*np.pi], elem_class=mock_facet)
    assert gas.max_elements_on_arc(300.) > 10
    n = len(gas.elem_pos)
    yz = np.empty((n, 2))
    for i, p in enumerate(gas.elem_pos):
        yz[i, :] = p[1:3, 3]
    phi = np.arctan2(yz[:, 1], yz[:, 0])
    ks, pvalue = kstest((phi + np.pi) / (2 * np.pi), 'uniform')
    assert pvalue > 0.3  # It's not exactly uniform because of finite size of elements.

def test_GAS_facets_on_radius():
    '''test distribution of elements on radius for d_r is non-integer multiple of d_element.'''
    myrowland = RowlandTorus(1000., 1000.)
    gas = GratingArrayStructure(myrowland, 60., [1000., 2000.], [300., 400.], elem_class=mock_facet)
    assert np.all(gas.distribute_elements_on_radius() == [320., 380.])
    gas.radius = [300., 340.]
    assert gas.distribute_elements_on_radius() == [320.]

def test_facet_rotation_via_facetargs():
    '''The numbers for the blaze are not realistic.'''
    gratingeff = uniform_efficiency_factory()
    mytorus = RowlandTorus(9e3/2, 9e3/2)
    mygas = GratingArrayStructure(mytorus, d_element=60., x_range=[5e3,1e4], radius=[538., 550.], elem_class=FlatGrating, elem_args={'zoom': 30, 'd':0.0002, 'order_selector': gratingeff})
    blaze = transforms3d.axangles.axangle2mat(np.array([0,1,0]), np.deg2rad(15.))
    mygascat = GratingArrayStructure(mytorus, d_element=60., x_range=[5e3,1e4], radius=[538., 550.], elem_class=FlatGrating, elem_args={'zoom': 30, 'orientation': blaze, 'd':0.0002, 'order_selector': gratingeff})
    assert np.allclose(np.rad2deg(np.arccos(np.dot(mygas.elements[0].geometry['e_x'][:3], mygascat.elements[0].geometry['e_x'][:3]))), 15.)

def test_persistent_facetargs():
    '''Make sure that facet_args is still intact after generating facets.

    This is important to allow tweaks of a single parameter and then to regenerate the facets.
    '''
    gratingeff = uniform_efficiency_factory()
    mytorus = RowlandTorus(9e4/2, 9e4/2)
    # id_col is automatically added in GAS is not present here.
    # So, pass in an id_col to make sure the comparison below will still work.
    facet_args = {'zoom': 30, 'd':0.0002, 'order_selector': gratingeff, 'id_col': 'facet'}
    mygas = GratingArrayStructure(mytorus, d_element=60., x_range=[5e4,1e5], radius=[5380., 5500.], elem_class=FlatGrating, elem_args=facet_args)
    assert mygas.elem_args == facet_args

def test_run_photons_through_gas():
    '''And check that they have the expected labels.

    No need to check here that the grating equation works - that's part of the grating tests/
    '''
    # Setup only.
    mysource = PointSource((30., 30.), flux=1., energy=1.)
    photons = mysource.generate_photons(1000)
    mypointing = FixedPointing(coords=(30, 30.))
    photons = mypointing.process_photons(photons)
    marxm = MarxMirror('./marxs/optics/hrma.par', position=np.array([0., 0, 0]))
    photons = marxm.process_photons(photons)
    gratingeff = uniform_efficiency_factory(1)
    facet_args = {'zoom': 30, 'd':0.0002, 'order_selector': gratingeff}
    mytorus = RowlandTorus(9e3/2, 9e3/2)

    # Now the real test with different input for facet_args

    for i in [1, 2, 3, 4]:
        if i == 1:
            kwargs = {}
            f = 'facet'
        elif i == 2:
            kwargs = {'id_col': 'qewr'}
            f = 'qewr'
        elif i == 3:
            facet_args['id_col'] = 'ttt'
            f = 'ttt'
            kwargs = {}
        elif i == 4:
            facet_args['id_col'] = 'uuu'
            f = 'uuu'
            kwargs = {'id_col': 'yyy'}

        mygas = GratingArrayStructure(mytorus, d_element=60., x_range=[5e3,1e4], radius=[538., 550.], elem_class=FlatGrating, elem_args=facet_args, **kwargs)

        p = mygas(photons.copy())
        indorder = np.isfinite(p['order'])
        indfacet = p[f] >=0
        assert np.all(indorder == indfacet)

        assert set(p['order'][indorder]) == set([-1, 0, 1])
        # -1 means no hit - passing between facets
        allfacets = set([-1]).union(set(np.arange(len(mygas.elements))))
        assert set(p[f]).issubset(allfacets)

def test_LinearCCDArray():
    '''Test an array in default position'''
    myrowland = RowlandTorus(10000., 10000.)
    # Along the way we normally would orient the detector.
    ccds = LinearCCDArray(myrowland, d_element=30., x_range=[0., 2000.],
                          radius=[-100., 100.], phi=0., elem_class=mock_facet)
    assert len(ccds.elements) == 7
    for e in ccds.elements:
        # For this test we don't care if z and e_z are parallel or antiparallel
        assert np.isclose(np.abs(np.dot([0, 0, 1, 0], e.geometry['e_z'])), 1.)
        assert (e.pos4d[0, 3] >= 0) and (e.pos4d[0, 3] < 1.)

    # center ccd is on the optical axis
    assert np.allclose(ccds.elements[3].geometry['e_y'], [0, 1, 0, 0])

def test_LinearCCDArray_rotatated():
    '''Test an array with different rotations.

    In this case, we rotate the Rowland torus by -30 deg and then
    check that the e_z vector is rotated 30 deg with respect to the
    coordinate system.
    '''
    pos4d = transforms3d.axangles.axangle2aff([1, 0, 0], np.deg2rad(-30))
    myrowland = RowlandTorus(10000., 10000., pos4d=pos4d)
    # Along the way we normally would orient the detector.
    ccds = LinearCCDArray(myrowland, d_element=30., x_range=[0., 2000.],
                          radius=[-100., 100.], phi=0., elem_class=mock_facet)
    assert len(ccds.elements) == 7
    for e in ccds.elements:
        assert np.isclose(np.dot([0, -0.8660254, 0.5], e.geometry['e_z'][:3]), 0, atol=1e-4)

def test_impossible_LinearCCDArray():
    '''The rotation is chosen such that all requested detector positions are
    INSIDE the rowland torus
    '''
    pos4d = transforms3d.axangles.axangle2aff([1, 0, 0], np.deg2rad(-30))
    myrowland = RowlandTorus(10000., 10000., pos4d=pos4d)
    with pytest.raises(ElementPlacementError) as e:
        ccds = LinearCCDArray(myrowland, d_element=30., x_range=[0., 2000.],
                              radius=[-100., 100.], phi=np.deg2rad(30.), elem_class=mock_facet)
    assert 'No intersection with Rowland' in str(e)
