import os

import numpy as np
import astropy
from transforms3d.axangles import axangle2mat
from transforms3d.affines import compose

from ..optics import FlatGrating, uniform_efficiency_factory
from ..simulator import Parallel

class HESS(Parallel):
    def __init__(self, **kwargs):
        path = os.path.dirname(__file__)
        self.hess = astropy.table.Table.read(os.path.join(path, 'HESSdesign.rdb'))
        '''The HESS datafile is commented very well inside the rdb file.
        Here, I just need to make a note about the relation of the coordinate systems:
        The vectors that define the facet edges are called x and y in the rdb file.
        In the MARXS global coordinate system these are y and z respectively, so
        uxf -> y and uyg -> z.

        Datafile from: http://space.mit.edu/HETG/hess/basic.html
        '''
        kwargs['elem_pos'] = None
        kwargs['elem_class'] = FlatGrating
        # Gratings are defined in order MEG, then HEG
        d = [4001.95 * 1e-7] * 192 + [2000.81 * 1e-7] * 144
        kwargs['elem_args'] = {'order_selector': uniform_efficiency_factory,
                               'd': d, 'name' : list(self.hess['hessloc'])}
        super(HESS, self).__init__(**kwargs)

    def calculate_elempos(self):
        '''Read position of facets from file.

        Based on the positions, other grating parameters are chosen that differ between
        HEG and MEG, e.g. grating constant.
        '''
        # take x,y,z components of various vectors and put in numpy arrays
        # Note: This makes (3, N) arrays as opposed to (N, 3) arrays used in other parts of MARXS.
        hess = self.hess

        cen = np.vstack([hess[s+'c'].data for s in 'xyz'])
        cen[0,:] +=(339.909-1.7)*25.4

        # All those vectors are already normalized
        facnorm = np.vstack([hess[s+'u'].data for s in 'xyz'])
        uxf = np.vstack([hess[s+'uxf'].data for s in 'xyz'])
        ul = np.vstack([hess[s+'ul'].data for s in 'xyz'])

        # Angle between y edge of grating and grove (edges not parallel to grooves)
        groove_angle = -np.arccos(np.einsum('ij,ij->j',ul, uxf))
        self.elem_args['groove_angle'] = list(groove_angle)
        zoom = np.array([1.035*25.4/2, 0.920*25.4/2, 1])

        pos4ds = []
        for i in range(facnorm.shape[1]):
            # Find angle and axis to rotate from old normal to facnorm
            axis1 = np.cross(np.array([1., 0., 0.]), facnorm[:, i])
            ang1 = np.arccos(facnorm[0, i])  # equivalent to: [1,0,0] * facnorm[:,1]
            rot1 = axangle2mat(axis1, ang1)
            # rotate round facnorm until y edges match
            # 1. Get rotated y axis
            yrot = np.dot(rot1, np.array([0., 1., 0.]))
            # 2. yrot and uxf should now both be in the plane of the facet.
            ang2 = np.arccos(np.dot(yrot, uxf[:, i]))
            # 3. Get rotation matrix around facet normal.
            rot2 = axangle2mat(facnorm[:, i], ang2)
            # Combine all the ingredients into 4d matrix
            rot = np.dot(rot2, rot1)
            pos4d = compose(cen[:, i], rot, zoom)
            pos4ds.append(pos4d)
        return pos4ds
