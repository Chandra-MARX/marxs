# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from transforms3d.axangles import axangle2mat
from transforms3d.affines import compose

from marxs.design.rowland import RectangularGrid
from marxs.simulator import Parallel, Sequence
from marxs.missions.mitsnl.catgrating import (CATL1L2Stack,
                                              catsupportbars)
from marxs.missions.arcus.load_csv import load_table
from marxs.missions.arcus.utils import config, id_num_offset
from marxs.missions.arcus.spo import zoom_to_cover_all_spos, rmid_spo


__all__ = [
           'CATfromMechanical',
           'CATWindow',
           'RegularGrid',
           ]


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
        pos4d = [compose(trans[i, :], rot[:, i, :].T, zoom[i])
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
        # currently, 'd' is the only parameter passed down that way
        # but e.g. orderselector could be treated the same way
        kwargs['elem_args']['elem_args'] = [{'d': d,
                                             'order_selector': self.conf['gratinggrid']['elem_args']['order_selector']} for d in d_grat]
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


class RegularGrid(Sequence):
    '''Regular grid of CAT gratings

    This regular grid of CAT gratings is useful in an early stage of
    development. Gratings are simply put down in a regular grid with
    no thought of mounting structures or optimizing the grating number
    under each SPO. While not the most efficient placement for a real
    mission, this is a useful design to get started before the details
    are worked out.
    '''
    def __init__(self, conf, channels=['1', '2', '1m', '2m'], **kwargs):

        elements = []
        temp, grid_width_y, grid_width_z = zoom_to_cover_all_spos(conf, .5, .5)
        rmid = rmid_spo(conf)

        blazemat = axangle2mat(np.array([0, 0, 1]),
                               np.deg2rad(-conf['blazeang']))
        blazematm = axangle2mat(np.array([0, 0, 1]),
                                np.deg2rad(conf['blazeang']))
        gratinggrid = conf['gratinggrid']

        for chan in channels:
            sig = 1 if '1' in chan else -1
            is_m = -1 if 'm' in chan else +1
            gratinggrid['rowland'] = conf['rowland_' + chan]
            b = blazematm if 'm' in chan else blazemat
            gratinggrid['elem_args']['orientation'] = b
            gratinggrid['normal_spec'] = conf['pos_opt_ax'][chan].copy()
            gratinggrid['guess_distance'] = is_m * (conf['rowland_central'].r + conf['rowland_central'].R)

            y_range = [-grid_width_y, grid_width_y]
            z_range = [sig * (rmid - grid_width_z),
                       sig * (rmid + grid_width_z)]
            z_range.sort()
            elements.append(RectangularGrid(y_range=y_range, z_range=z_range,
                                            id_num_offset=id_num_offset[chan],
                                            **gratinggrid))
        elements.extend([catsupportbars])
        super().__init__(elements=elements, **kwargs)
