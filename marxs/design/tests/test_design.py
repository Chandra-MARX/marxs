# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import kstest
import transforms3d
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename

from ..rowland import (RowlandTorus, GratingArrayStructure,
                       RectangularGrid,
                       find_radius_of_photon_shell, design_tilted_torus,
                       CircularMeshGrid,
                       double_rowland_from_channel_distance,
                       add_offset_double_rowland_channels,
                       )
from ...optics.base import FlatOpticalElement
from ...source import PointSource, FixedPointing
from ...optics import MarxMirror, OrderSelector, FlatGrating
from ...math.utils import h2e, xyz2zxy

class mock_facet(FlatOpticalElement):
    '''Lightweight class with no functionality for tests.'''
    pass

def test_radius_of_photon_shell():
    mysource = PointSource(coords=SkyCoord(30., 30., unit="deg"))
    photons = mysource.generate_photons(1. * u.ks)
    mypointing = FixedPointing(coords=SkyCoord(30, 30., unit='deg'))
    photons = mypointing.process_photons(photons)
    marxm = MarxMirror(get_pkg_data_filename('hrma.par', package='marxs.optics'), position=np.array([0., 0, 0]))
    photons = marxm(photons)
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
    assert r == pytest.approx(5.0068617299896054)
    assert R == pytest.approx(4.9794336175561327)
    assert pos4d[0,3] == pytest.approx(0.027390523158633273)
    # The z axis should be unchanged
    assert np.allclose(pos4d[[0,1],2], 0)

def test_design_tilted_torus_negative_angles():
    '''Tilt the torus the other way. Should be same geometry in general.'''
    R, r, pos4d = design_tilted_torus(10, np.deg2rad(3), np.deg2rad(6))
    Rn, rn, pos4dn = design_tilted_torus(10, np.deg2rad(-3), np.deg2rad(-6))
    assert R == pytest.approx(Rn)
    assert r == pytest.approx(rn)
    assert pos4d[0,3] == pytest.approx(pos4dn[0, 3])
    assert pos4d[2, 3] == 0 # torus in xy plane
    assert pos4dn[2, 3] == 0
    assert pos4d[1, 3] == pytest.approx(-pos4dn[1, 3])

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
    '''solve_quartic helps to find points on the torus.

    This test could make use of the hypothesis testing package.
    '''
    mytorus = RowlandTorus(5., 4.)
    xyzw = mytorus.parametric([.34, 1.23], [.4, 4.5])

    for i in [0, 1]:
        # generate random direction in homogeneous coordinates
        direction = np.random.rand(4) - 0.5
        direction[3] = 0
        assert np.allclose(xyzw[i, :],
                            mytorus.solve_quartic(origin=xyzw[i, :] + 0.1 * direction,
                                                  v=direction))


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


def test_torus_xyzw2parametric():
    '''Theta, phi coordinates for torus should round-trip to xyz'''
    R = 2.
    r = 1.

    angle = np.arange(0, 2 * np.pi, 0.1)
    phi, theta = np.meshgrid(angle, angle)
    phi = phi.flatten()
    theta = theta.flatten()
    mytorus = RowlandTorus(R=R, r=r)

    xyzw = mytorus.parametric(theta, phi)
    t, p = mytorus.xyzw2parametric(xyzw)
    assert np.allclose(np.mod(theta, 2. * np.pi), np.mod(t, 2. * np.pi))
    assert np.allclose(np.mod(phi, 2. * np.pi), np.mod(p, 2. * np.pi))


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
    t1 = mytorus.parametric(theta + 0.00001, phi)
    p1 = mytorus.parametric(theta, phi + 0.00001)
    t0 = mytorus.parametric(theta - 0.00001, phi)
    p0 = mytorus.parametric(theta, phi - 0.00001)

    vec_normal = h2e(mytorus.normal(xyz))

    vec_delta_theta = h2e(t1) - h2e(t0)
    vec_delta_theta = vec_delta_theta  / np.linalg.norm(vec_delta_theta, axis=1)[:, None]
    vec_delta_phi = h2e(p1) - h2e(p0)
    vec_delta_phi = vec_delta_phi  / np.linalg.norm(vec_delta_phi, axis=1)[:, None]

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
    gas = GratingArrayStructure(rowland=myrowland,
                                d_element=[30., 30.],
                                radius=[300., 600.],
                                phi=[-0.2*np.pi, 0.2*np.pi], elem_class=mock_facet)
    assert gas.max_elements_on_arc(300.) == 12
    angles = gas.distribute_elements_on_arc(315.) % (2. * np.pi)
    # This is a wrap-around case. Hard to test in general, but here I know the numbers
    assert np.alltrue((angles < 0.2 * np.pi) | (angles > 1.8 * np.pi))
    assert gas.max_elements_on_radius(gas.radius) == 10
    assert np.all(gas.distribute_elements_on_radius() == np.arange(315., 600., 30.))
    assert len(gas.elem_pos) == 177
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

def test_GAS_multipleradii():
    '''Radius can be a list of many pairs.'''
    myrowland = RowlandTorus(1000., 1000.)
    gas1 = GratingArrayStructure(rowland=myrowland, d_element=[60., 60.],
                                 radius=[300., 400.], elem_class=mock_facet)
    gas2 = GratingArrayStructure(rowland=myrowland, d_element=[60., 60.],
                                 radius=[500., 540.], elem_class=mock_facet)
    gas = GratingArrayStructure(rowland=myrowland, d_element=[60., 60.],
                                radius=[300., 400., 500., 540.],
                                elem_class=mock_facet)

    r1 = gas1.distribute_elements_on_radius()
    r2 = gas2.distribute_elements_on_radius()
    r = gas.distribute_elements_on_radius()
    assert np.all(np.in1d(r, np.union1d(r1, r2)))


def test_GratingArrayStructure_2pi():
    '''test that delta_phi = 2 pi means "full circle" and not 0
    '''
    myrowland = RowlandTorus(10000., 10000.)
    gas = GratingArrayStructure(rowland=myrowland, d_element=[30., 30.],
                                radius=[300., 600.],
                                phi=[0, 2*np.pi], elem_class=mock_facet)
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
    gas = GratingArrayStructure(rowland=myrowland, d_element=[60., 60.],
                                radius=[300., 400.], elem_class=mock_facet)
    assert np.all(gas.distribute_elements_on_radius() == [320., 380.])
    gas.radius = [300., 340.]
    assert gas.distribute_elements_on_radius() == [320.]

def test_facet_rotation_via_facetargs():
    '''The numbers for the blaze are not realistic.'''
    gratingeff = OrderSelector(np.arange(-3, 4))
    mytorus = RowlandTorus(9e3/2, 9e3/2)
    mygas = GratingArrayStructure(rowland=mytorus, d_element=[60., 60.],
                                  radius=[538., 550.], elem_class=FlatGrating,
                                  elem_args={'zoom': 30, 'd':0.0002, 'order_selector': gratingeff})
    blaze = transforms3d.axangles.axangle2mat(np.array([0,1,0]), np.deg2rad(15.))
    mygascat = GratingArrayStructure(rowland=mytorus, d_element=[60., 60.],
                                     radius=[538., 550.], elem_class=FlatGrating,
                                     elem_args={'zoom': 30, 'orientation': blaze, 'd':0.0002, 'order_selector': gratingeff})
    assert np.allclose(np.rad2deg(np.arccos(np.dot(mygas.elements[0].geometry['e_x'][:3], mygascat.elements[0].geometry['e_x'][:3]))), 15.)

def test_persistent_facetargs():
    '''Make sure that facet_args is still intact after generating facets.

    This is important to allow tweaks of a single parameter and then to regenerate the facets.
    '''
    gratingeff = OrderSelector(np.arange(-3, 4))
    mytorus = RowlandTorus(9e4/2, 9e4/2)
    # id_col is automatically added in GAS is not present here.
    # So, pass in an id_col to make sure the comparison below will still work.
    facet_args = {'zoom': 30, 'd':0.0002, 'order_selector': gratingeff, 'id_col': 'facet'}
    mygas = GratingArrayStructure(rowland=mytorus, d_element=[60., 60.],
                                  radius=[5380., 5500.], elem_class=FlatGrating,
                                  elem_args=facet_args)
    assert mygas.elem_args == facet_args

def test_run_photons_through_gas():
    '''And check that they have the expected labels.

    No need to check here that the grating equation works - that's part of the grating tests.
    '''
    # Setup only.
    mysource = PointSource(coords=SkyCoord(30., 30., unit="deg"))
    photons = mysource.generate_photons(1. * u.ks)
    mypointing = FixedPointing(coords=SkyCoord(30, 30., unit='deg'))
    photons = mypointing.process_photons(photons)
    marxm = MarxMirror(get_pkg_data_filename('hrma.par', package='marxs.optics'), position=np.array([0., 0, 0]))
    photons = marxm(photons)
    gratingeff = OrderSelector([-1, 0, 1])
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

        mygas = GratingArrayStructure(rowland=mytorus, d_element=[60., 60.],
                                      radius=[538., 550.], elem_class=FlatGrating,
                                      elem_args=facet_args, **kwargs)

        p = mygas(photons.copy())
        indorder = np.isfinite(p['order'])
        indfacet = p[f] >=0
        assert np.all(indorder == indfacet)

        assert set(p['order'][indorder]) == set([-1, 0, 1])
        # -1 means no hit - passing between facets
        allfacets = set([-1]).union(set(np.arange(len(mygas.elements))))
        assert set(p[f]).issubset(allfacets)


def test_RectangularGrid_rotated():
    '''Test an array with different rotations.

    In this case, we rotate the Rowland torus by -30 deg and then
    check that the e_z vector is rotated 30 deg with respect to the
    coordinate system.
    '''
    pos4d = transforms3d.axangles.axangle2aff([1, 0, 0], np.deg2rad(-30))
    myrowland = RowlandTorus(10000., 10000., pos4d=pos4d)
    # Along the way we normally would orient the detector.
    ccds = RectangularGrid(rowland=myrowland, d_element=[30., 30.],
                           y_range=[-100, 100],
                           elem_class=mock_facet,
                           guess_distance=100)
    assert len(ccds.elements) == 7
    for e in ccds.elements:
        assert np.isclose(np.dot([0, -0.8660254, 0.5], e.geometry['e_z'][:3]), 0, atol=1e-4)


def test_RectangularGridone():
    '''Test an array of one element in default position'''
    myrowland = RowlandTorus(10000., 10000.)
    # Along the way we normally would orient the detector.
    ccds = RectangularGrid(rowland=myrowland, d_element=[300., 300.],
                           y_range=[-1, 1],
                           elem_class=mock_facet,
                           guess_distance=25)
    assert len(ccds.elements) == 1
    # multiply because of pointing inward vs. outward
    assert np.allclose(ccds.elements[0].pos4d, np.eye(4))


def test_RegtangularGrid_offcenter_unaffected():
    '''Make sure the coordinate system of the xyz_range is in global coordinates.
    In this test, we check the z position, which in unchanged because the torus
    center is moved only in the y direction.'''
    R, r, pos4d = design_tilted_torus(5000, 0.07, 0.15)
    rowland = RowlandTorus(R, r, pos4d=pos4d)
    ccds = RectangularGrid(rowland=rowland, d_element=[5., 5.],
                           z_range=[10, 20],
                           y_range=[600, 615],
                           elem_class=mock_facet,
                           guess_distance=25)

    for e in ccds.elements:
        assert (e.geometry['center'][2] > 10) and (e.geometry['center'][2] < 20)


def test_RegtangularGrid_offcenter_affected():
    '''Make sure the coordinate system of the xyz_range is in global coordinates.
    In this test, we check the y position.'''
    R, r, pos4d = design_tilted_torus(50000, 0.007, 0.015)
    rowland = RowlandTorus(R, r, pos4d=pos4d)
    ccds = RectangularGrid(rowland=rowland, d_element=[50., 50.],
                           y_range=[600, 800],
                           elem_class=mock_facet,
                           guess_distance=25)
    y_offset = rowland.geometry['center'][1]
    for e in ccds.elements:
        assert (e.geometry['center'][1] > 600 + y_offset) and \
               (e.geometry['center'][1] < 800 + y_offset)



def test_nojumpsinCCDorientation():
    '''Regression test: CCD orientation should change smoothly along the circle.'''
    R, r, pos4d = design_tilted_torus(12e3, 0.07, 0.15)
    rowland = RowlandTorus(R, r, pos4d=pos4d)
    det = RectangularGrid(rowland=rowland,
                             elem_class=mock_facet,
                             d_element=[49.652, 49.652],
                             y_range=[-200, +200],
                             guess_distance=25.)

    # If all is right, this will vary very smoothly along the circle.
    xcomponent = np.array([e.geometry['e_y'][0] for e in det.elements])
    xcompdiff = np.diff(xcomponent)
    assert (np.max(np.abs(xcompdiff)) / np.median(np.abs(xcompdiff))) < 1.1


def test_circularMeshGrid():
    '''Check MeshGrid against a hand-checked and thus known-good solution'''
    myrowland = RowlandTorus(10000., 10000.)
    gas = CircularMeshGrid(rowland=myrowland,
                           radius=(100., 300.),
                           d_element=(30., 50),
                           elem_class=mock_facet,
                           optimize_axis=np.array([1, 0, 0, 0]),
                           parallel_spec=np.array([0., 1., 0., 0.]),
                           normal_spec=np.array([1, 0, 0, 1]))
    assert len(gas.elements) == 178
    poslist = np.stack(gas.elem_pos)
    assert np.all(poslist[:, 0, 3] > 19000)
    assert np.all(poslist[:, 0, 3] < 20000)
    assert np.all(np.abs(poslist[:, 1, 3] < 300))
    assert np.all(np.abs(poslist[:, 2, 3] < 300))

def test_circularMeshGrid_rotated():
    '''Check MeshGrid against a hand-checked and thus known-good solution.
    Repeats previous test but for a different axes'''
    myrowland = RowlandTorus(10000., 10000.)
    myrowland.pos4d = xyz2zxy @ myrowland.pos4d
    gas = CircularMeshGrid(rowland=myrowland,
                           radius=(100., 300.),
                           d_element=(30., 50),
                           elem_class=mock_facet,
                           optimize_axis=np.array([0, 0, 1, 0]),
                           parallel_spec=np.array([0., 1., 0., 0.]),
                           normal_spec=np.array([0, 1, 0, 1]))
    assert len(gas.elements) == 178
    poslist = np.stack(gas.elem_pos)
    assert np.all(poslist[:, 2, 3] > 19000)
    assert np.all(poslist[:, 2, 3] < 20000)
    assert np.all(np.abs(poslist[:, 0, 3] < 300))
    assert np.all(np.abs(poslist[:, 1, 3] < 300))


def test_double_rowland_from_channel_distance():
    '''Check that the double torus is in the right position.
    The numbers are hand-checked and thus known-good.
    '''
    geom = double_rowland_from_channel_distance(600., 6000, 12000)
    assert geom['pos_opt_ax']['1'] == pytest.approx([0, -300, 0, 1])
    assert geom['pos_opt_ax']['1m'] == pytest.approx([0, 300, 0, 1])

    assert geom['rowland_detector'].R == pytest.approx(6000)
    assert geom['rowland_detector'].r == pytest.approx(6007.495318350236)
    assert geom['rowland_central'].geometry['center'] == pytest.approx([7.48596673, 599.62570166, 0.,  1.])


def test_offset_double_rowland_channels():
    '''Split the double torus into two and offset all four channels.

    Checked against the version of the code that's been in use for years
    for Arcus.
    '''
    geom = double_rowland_from_channel_distance(600., 6000, 12000)
    add_offset_double_rowland_channels(geom,
                                   offsets={'1': [0, -2.5, -7.5],
                                            '1m': [0, -2.5, -2.5],
                                            '2': [0, 2.5, 2.5],
                                            '2m': [0, 2.5, 12.5],
                                            })
    assert geom['pos_opt_ax']['2m'] == pytest.approx([0, 302.5, 12.5, 1. ])
    assert geom['rowland_1'].geometry['center'] == pytest.approx([7.48596673, 297.12570166, -7.5, 1.])


def test_rowlands_are_oriented_the_same_way():
    """When there are two rowland tori, they should be oriented the same way."""
    double = double_rowland_from_channel_distance(100, 500, 1000.0)
    add_offset_double_rowland_channels(
        double, offsets={"1": [0, -5, 0], "1m": [0, +5, 0]}
    )
    assert_allclose(
        double["rowland_central"].parametric(0, 0),
        double["rowland_central_m"].parametric(0, 0),
    )
    # The "top" points in the same direction. Of course, they are offset from each other
    # so the numbers are not exactly the same.
    assert_allclose(double["rowland_1"].parametric(0, 0), [1000.0, -55.0, 0.0, 1.0])
    assert_allclose(double["rowland_1m"].parametric(0, 0), [1000.0, 55.0, 0.0, 1.0])