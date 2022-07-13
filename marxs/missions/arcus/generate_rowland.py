# Licensed under GPL version 3 - see LICENSE.rst
'''This module has functions to make up a conf dictionary that
specified all the optical parameters for Arcus. Several of the
parameters depend on each other (e.g. the spaceing between the CTS
gratings and the focal plane depend on the focal lengths and the
spacing between SPOs and CAT gratings), so these functions derive the
remaining parameters from specific input variables.

It would be cool to have a system that could take any combination of
parameters and derive the remaining ones, but for now, I just handcode
those functions that I used.

'''

import numpy as np
from marxs.design.rowland import design_tilted_torus, RowlandTorus
from marxs.math.utils import xyz2zxy


def derive_rowland_and_shiftmatrix(blazeang, max_z_torus):
    '''This is what I used initially. Some parameters of choice are hardcoded.
    Essentially, this fixes the blaze angle and then makes tilted Rowland tori
    that are tilted just a little more than the glaze angle.
    '''

    out = {'blazeang': blazeang,  # Blaze angle in degrees
           'offset_spectra': 5.,  # Offset of each of the two spectra from CCD center
           'alpha': np.deg2rad(2.2 * 1.91),
           'beta': np.deg2rad(4.4 * 1.91),
           'max_z_torus': max_z_torus}

    R, r, pos4d = design_tilted_torus(out['max_z_torus'],
                                      out['alpha'],
                                      out['beta'])
    out['rowland_central'] = RowlandTorus(R, r, pos4d=pos4d)
    out['rowland_central'].pos4d = np.dot(xyz2zxy,
                                          out['rowland_central'].pos4d)

    # Now offset that Rowland torus in a z axis by a few mm.
    # Shift is measured from a focal point that hits the center of the CCD strip.
    out['shift_optical_axis_1'] = np.eye(4)
    out['shift_optical_axis_1'][1, 3] = - out['offset_spectra']

    out['rowland'] = RowlandTorus(R, r, pos4d=pos4d)
    out['rowland'].pos4d = np.dot(xyz2zxy, out['rowland'].pos4d)
    out['rowland'].pos4d = np.dot(out['shift_optical_axis_1'],
                                  out['rowland'].pos4d)

    Rm, rm, pos4dm = design_tilted_torus(out['max_z_torus'],
                                         - out['alpha'],
                                         - out['beta'])
    out['rowlandm'] = RowlandTorus(Rm, rm, pos4d=pos4dm)
    out['rowlandm'].pos4d = np.dot(xyz2zxy, out['rowlandm'].pos4d)

    out['d'] = r * np.sin(out['alpha'])
    # Relative to origin in the center of the CCD strip
    out['shift_optical_axis_2'] = np.eye(4)
    out['shift_optical_axis_2'][0, 3] = 2. * out['d']
    out['shift_optical_axis_2'][1, 3] = + out['offset_spectra']

    # Optical axis 2 relative to optical axis 1
    out['shift_optical_axis_12'] = np.dot(np.linalg.inv(out['shift_optical_axis_1']),
                                          out['shift_optical_axis_2'])

    out['rowlandm'].pos4d = np.dot(out['shift_optical_axis_2'],
                                   out['rowlandm'].pos4d)

    return out


def make_rowland_from_d_BF_R_f(d_BF, R, f=11880.):
    '''This prescription is borne out of the engineering needs where it turns out
    that the spacing between channels d_BF is actually an important parameter
    so it makes sense to use that here to parameterize the torus, too.

    f is the distance between a theoerical on-axis CAT grating and the focal
    point.  It's default is a little smaller than 12 m, to leave space to mount
    the SPOs.

    '''
    d = 0.5 * d_BF
    r = 0.5 * np.sqrt(f**2 + d_BF**2)
    alpha = np.arctan2(d_BF, f)
    pos = [(r + R) * np.sin(alpha),
           0, f - (r + R) * np.cos(alpha)]
    orient = [[-np.sin(alpha), np.cos(alpha), 0],
              [0., 0., 1],
              [np.cos(alpha), np.sin(alpha), 0]]

    posm = [(r + R) * np.sin(-alpha),
            0, f - (r + R) * np.cos(-alpha)]

    orientm = [[-np.sin(alpha), -np.cos(alpha), 0],
               [0., 0., 1],
               [-np.cos(alpha), np.sin(alpha), 0]]

    geometry = {'d_BF': d_BF,
                'd': d,
                'f': f,
                'rowland_central': RowlandTorus(R=R, r=r, position=pos,
                                                orientation=orient),
                'rowland_central_m': RowlandTorus(R=R, r=r, position=posm,
                                                  orientation=orientm)}
    geometry['pos_opt_ax'] = {'1': np.array([-d - 2.5, -7.5, 0., 1]),
                              '1m': np.array([d - 2.5, -2.5, 0, 1]),
                              '2': np.array([-d + 2.5, +2.5, 0, 1]),
                              '2m': np.array([d + 2.5, 7.5, 0, 1])}
    geometry['pos_det_rowland'] = np.array([-d, 0, 0, 1])

    # Now offset that Rowland torus in a z axis by a few mm.
    # Shift is measured from point on symmetry plane.
    for channel in geometry['pos_opt_ax']:
        name = 'shift_optical_axis_' + channel
        geometry[name] = np.eye(4)
        geometry[name][:, 3] = geometry['pos_opt_ax'][channel][:]
        if channel in ['1', '2']:
            base_rowland = 'rowland_central'
        elif channel in ['1m', '2m']:
            base_rowland = 'rowland_central_m'
        namer = 'rowland_' + channel
        geometry[namer] = RowlandTorus(R, r,
                                       pos4d=geometry[base_rowland].pos4d)
        geometry[namer].pos4d = np.dot(geometry[name],
                                       geometry[namer].pos4d)

    geometry['shift_det_rowland'] = np.eye(4)
    geometry['shift_det_rowland'][:, 3] = geometry['pos_det_rowland']
    geometry['rowland_detector'] = RowlandTorus(R, r,
                                                pos4d=geometry['rowland_central'].pos4d)
    geometry['rowland_detector'].pos4d = np.dot(geometry['shift_det_rowland'],
                                                geometry['rowland_detector'].pos4d)


    return geometry
