import numpy as np
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from transforms3d.affines import compose

from ...optics import FlatGrating, uniform_efficiency_factory
from ...simulator import Parallel

class HETG(Parallel):

    id_col = 'facet'

    def __init__(self, **kwargs):
        self.hess = Table.read(get_pkg_data_filename('HESSdesign.rdb'))
        '''The HESS datafile is commented very well inside the rdb file.
        Here, I just need to make a note about the relation of the coordinate systems:
        The vectors that define the facet edges are called x and y in the rdb file.
        In the MARXS global coordinate system these are y and z respectively, so
        uxf -> y and uyg -> z.

        Datafile from: http://space.mit.edu/HETG/hess/basic.html
        '''
        kwargs['elem_pos'] = self.calculate_elempos()
        kwargs['elem_class'] = FlatGrating
        # Gratings are defined in order MEG, then HEG
        d = [4001.95 * 1e-7] * 192 + [2000.81 * 1e-7] * 144
        ul = np.vstack([self.hess[s+'ul'].data for s in 'xyz'])
        uyf = np.vstack([self.hess[s + 'uyf'].data for s in 'xyz'])
        uxf = np.vstack([self.hess[s + 'uxf'].data for s in 'xyz'])
        groove_ang = -np.arctan2(np.einsum('i...,i...->...', ul, uxf),
                                 np.einsum('i...,i...->...', ul, uyf))
        kwargs['elem_args'] = {'order_selector': uniform_efficiency_factory(),
                               'd': d, 'name': list(self.hess['hessloc']),
                               'groove_angle': groove_ang.tolist()}
        super(HETG, self).__init__(**kwargs)

    def calculate_elempos(self):
        '''Read position of facets from file.

        Based on the positions, other grating parameters are chosen that differ between
        HEG and MEG, e.g. grating constant.
        '''
        # take x,y,z components of various vectors and put in numpy arrays
        # Note: This makes (3, N) arrays as opposed to (N, 3) arrays used in other parts of MARXS.
        # Note: Sometimes an angle has a minus. Beware of active / passive rotations.
        hess = self.hess

        cen = np.vstack([hess[s+'c'].data for s in 'xyz'])
        cen[0, :] += (339.909 - 1.7) * 25.4

        # All those vectors are already normalized
        facnorm = np.vstack([hess[s + 'u'].data for s in 'xyz'])
        uxf = np.vstack([hess[s + 'uxf'].data for s in 'xyz'])
        uyf = np.vstack([hess[s + 'uyf'].data for s in 'xyz'])

        pos4ds = []
        for i in range(facnorm.shape[1]):
            if int(self.hess['hessloc'][i][0]) <= 3:
                # Size of MEG facets
                zoom = np.array([1, 1.035*25.4/2, 1.035*25.4/2])
            else:
                # Size of HEG facets
                zoom = np.array([1, 0.920*25.4/2, 0.920*25.4/2])

            rot = np.eye(3)
            rot[:, 0] = facnorm[:, i]
            rot[:, 1] = uxf[:, i]
            rot[:, 2] = uyf[:, i]
            pos4d = compose(cen[:, i], rot, zoom)
            pos4ds.append(pos4d)
        return pos4ds
