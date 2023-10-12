# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np

from transforms3d.euler import euler2mat
from transforms3d.affines import compose
from marxs.optics import OpticalElement
from marxs.simulator import Parallel
from marxs.math.geometry import Cylinder


class Rod(OpticalElement):
    '''X-axis of the rod is the cylinder axis (x-zoom gives half-length)
    y zoom and z zoom have to be the same (no elliptical cylinders)
    '''
    outcol = 'hitrod'
    display = {'shape': 'cylinder', 'color': 'black'}

    default_geometry = Cylinder

    def process_photons(self, photons, intersect, interpos, intercoos):
        if self.outcol not in photons.colnames:
            photons[self.outcol] = False
        photons[self.outcol][intersect] = True
        photons['probability'][intersect] = 0.
        return photons


class ThreeSidedBoom(Parallel):
    '''Three-sided boom with dimensions similar to the Arcus proposal.

    Parameters
    ----------
    boom_dims : dict (optional)
        Entries in `boom_dict` will overwrite class attributes. It's meant
        to set different dimensions for the boom.
    '''
    l_longeron = 1.08 * 1e3
    l_batten = 1.6 * 1e3
    l_diag = np.sqrt(l_longeron**2 + l_batten**2)
    d_longeron = 10.2
    d_batten = 8.
    d_diag = 2.
    # If the bottom of a boom does not contribute to the absorption
    # setting start_bay to a larger number can shorten the computation time.
    start_bay = 0
    n_bays = 10
    n_sides = 3

    def __init__(self, **kwargs):
        boom_dims = kwargs.pop('boom_dimensions', {})
        for k in boom_dims:
            if hasattr(self, k):
                setattr(self, k, boom_dims[k])
            else:
                raise ValueError('{} has not attribute {}'.format(self.name, k))
        kwargs['elem_pos'] = self.pos_spec()
        kwargs['elem_class'] = Rod
        kwargs['elem_args'] = {}
        super(ThreeSidedBoom, self).__init__(**kwargs)

    def one_window(self):
        '''Positions for rods on one side of one bay

        Returns
        -------
        pos4d : list
            List of pos4d matrices
        '''
        # longeron
        zoom = [self.l_longeron / 2, self.d_longeron / 2, self.d_longeron / 2]
        trans = [self.l_longeron / 2, self.l_batten / 3**0.5, 0.]
        rot = np.eye(3)
        pos4d = [compose(trans, rot, zoom)]

        # batten
        zoom = [self.l_batten / 2, self.d_batten / 2, self.d_batten / 2]
        trans = [0., self.l_batten / 4. / 3**0.5, self.l_batten / 4]
        rot = euler2mat(np.pi / 2, - np.pi / 6, 0, 'szxz')
        pos4d.append(compose(trans, rot, zoom))

        # diagonal1
        zoom = [self.l_diag / 2., self.d_diag / 2., self.d_diag / 2.]
        trans[0] = self.l_longeron / 2.
        rot = euler2mat(np.deg2rad(90. - 34.03), -np.pi / 6, 0, 'szxz')
        pos4d.append(compose(trans, rot, zoom))

        # diagonal2
        rot = euler2mat(np.deg2rad(90. + 34.03), -np.pi / 6, 0, 'szxz')
        pos4d.append(compose(trans, rot, zoom))
        return pos4d

    def pos_spec(self):
        '''Calculate pos4d matrices for each element on the boom'''

        fullboompos4d = []
        for h in range(self.start_bay, self.n_bays):
            trans = [h * self.l_longeron, 0, 0]
            for i in range(self.n_sides):
                rot = euler2mat((i * 2) * np.pi / self.n_sides, 0, 0, 'sxyz')
                affmat = compose(trans, rot, np.ones(3))
                for p in self.one_window():
                    fullboompos4d.append(np.dot(affmat, p))

        return fullboompos4d


class FourSidedBoom(ThreeSidedBoom):
    '''Four-sided boom with dimensions according to the ARCUS proposal.

    Parameters
    ----------
    boom_dims : dict (optional)
        Entries in `boom_dict` will overwrite class attributes. It's meant
        to set different dimensions for the boom.
    '''
    l_longeron = 1350.5
    l_batten = 1308.
    l_diag = np.sqrt(l_longeron**2 + l_batten**2)
    d_longeron = 10.2
    d_batten = 8.43
    d_diag = 1.85
    n_bays = 8
    n_sides = 4

    def one_window(self):
        '''Calculate pos4d matrices for each element on the boom'''
        # longeron
        zoom = [self.l_longeron / 2, self.d_longeron / 2, self.d_longeron / 2]
        trans = [self.l_longeron / 2, self.l_batten / 2, self.l_batten / 2]
        rot = np.eye(3)
        pos4d = [compose(trans, rot, zoom)]

        # batten
        zoom = [self.l_batten / 2, self.d_batten / 2, self.d_batten / 2]
        trans = [0., self.l_batten / 2, 0]
        rot = euler2mat(np.pi / 2, 0, 0, 'syxz')
        pos4d.append(compose(trans, rot, zoom))

        # diagonal1
        zoom = [self.l_diag / 2., self.d_diag / 2., self.d_diag / 2.]
        trans = [self.l_longeron / 2., self.l_batten / 2, 0.]
        ang = np.arcsin(self.l_batten / self.l_diag)
        rot = euler2mat(ang, 0, 0, 'syxz')
        pos4d.append(compose(trans, rot, zoom))

        # diagonal2
        rot = euler2mat(- ang, 0, 0, 'syxz')
        pos4d.append(compose(trans, rot, zoom))

        return pos4d
