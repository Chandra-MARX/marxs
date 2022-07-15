# Licensed under GPL version 3 - see LICENSE.rst
import os

import numpy as np
import transforms3d
from marxs.optics import OpticalElement
from marxs.simulator import Parallel, ParallelCalculated
from marxs.missions.mitsnl.catgrating import (InterpolateEfficiencyTable,
                                              CATL1L2Stack)


from .load_csv import load_table
from .utils import config


globalorderselector = InterpolateEfficiencyTable(
    os.path.join(config['data']['caldb_inputdata'],
                 'gratings', 'efficiency.csv'))
'''Global instance of an order selector to use in all CAT gratings.

As long as the efficiency table is the same for all CAT gratings, it makes
sense to define that globaly. If every grating had its own independent
order selector, we would have to read the selector file a few hundred times.
'''


class CATfromMechanical(Parallel):
    '''A collection of diffraction gratings on the Rowland torus.

    After any of the `elem_pos`, `elem_uncertainty` or
    `uncertainty` is changed, `generate_elements` needs to be
    called to regenerate the facets on the GAS.
    '''

    id_col = 'window'

    def stack(self, name):
        return np.vstack([self.data[name + 'X'].data,
                          self.data[name + 'Y'].data,
                          self.data[name + 'Z'].data]).T

    def __init__(self, **kwargs):
        self.channel = kwargs.pop('channel')
        self.conf = kwargs.pop('conf')
        self.data = load_table('gratings', 'facets')

        zoom = [[1, row['xsize'] / 2, row['ysize'] / 2] for row in self.data]
        trans = self.stack('')
        rot = np.stack([self.stack('NormN'),
                        self.stack('DispN'),
                        self.stack('GBarN')])
        pos4d = [transforms3d.affines.compose(trans[i, :],
                                              rot[:, i, :].T,
                                              zoom[i])
                 for i in range(len(self.data))]

        mirr = np.eye(4)
        if 'm' in self.channel:
            mirr[0, 0] = 1
        else:
            mirr[0, 0] = -1
        if '2' in self.channel:
            mirr[1, 1] = -1

        pos4d = [np.dot(mirr, p) for p in pos4d]

        # Shift pos4d from focal point to center of grating petal
        # Ignore rotations at this point
        pos4d = np.array(pos4d)
        centerpos = np.eye(4)
        centerpos[:, 3] = pos4d.mean(axis=0)[:, 3]
        centerpos_inv = np.linalg.inv(centerpos)
        kwargs['pos4d'] = np.dot(centerpos, kwargs['pos4d'])
        pos4d = np.array([np.dot(centerpos_inv, p) for p in pos4d])

        windowpos = []
        gratingpos = []
        id_start = []
        d_grat = []

        for i in sorted(set(self.data['SPO_MM_num'])):
            ind = self.data['SPO_MM_num'] == i
            winpos = np.eye(4)
            winpos[:, 3] = pos4d[ind, :, :].mean(axis=0)[:, 3]
            winpos_inv = np.linalg.inv(winpos)
            windowpos.append(winpos)
            id_start.append(kwargs.get('id_num_offset', 0) +
                            self.data['facet_num'][ind][0])
            grat_pos = [np.dot(winpos_inv, pos4d[j, :, :])
                        for j in ind.nonzero()[0]]
            gratingpos.append(grat_pos)
            d_grat.append(list(self.data['period'][ind]))

        kwargs['elem_pos'] = windowpos
        kwargs['elem_class'] = CATWindow
        kwargs['elem_args'] = {}
        # This is the elem_args for the second level (the CAT gratings).
        # We need a list of dicts. Each window will then get one dict
        # which itself is a dict of lists
        # currenty, 'd' is the only parameter passed down that way
        # but e.g. orderselector could be treated the same way
        kwargs['elem_args']['elem_args'] = [{'d': d,
                                             'order_selector': globalorderselector} for d in d_grat]
        kwargs['elem_args']['elem_pos'] = gratingpos
        kwargs['elem_args']['id_num_offset'] = id_start
        super(CATfromMechanical, self).__init__(**kwargs)

    def generate_elements(self):
        super(CATfromMechanical, self).generate_elements()
        for e in self.elements:
            e.generate_elements()


class CATWindow(Parallel):

    id_col = 'facet'

    def __init__(self, **kwargs):
        kwargs['id_col'] = self.id_col
        kwargs['elem_class'] = CATL1L2Stack
        super().__init__(**kwargs)
