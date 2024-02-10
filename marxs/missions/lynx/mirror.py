# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d, RectBivariateSpline
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename

from marxs import optics
from marxs.simulator import Sequence
from marxs.base import SimulationSequenceElement

from marxs.math.utils import e2h, h2e, norm_vector
from marxs.math.polarization import parallel_transport


class MetaShellAperture(optics.MultiAperture):

    def __init__(self, conf, **kwargs):
        metashellgeometry = conf['metashellgeometry']
        elements = [optics.CircleAperture(position=[conf['focallength'] + 200, 0, 0],
                                          zoom=[1, shell['r_outer'], shell['r_outer']],
                                          r_inner=shell['r_inner'],
                                          id_num=shell['Metashell Serial Number'])
                    for shell in metashellgeometry]
        kwargs['elements'] = elements
        kwargs['id_col'] = 'shell'
        super().__init__(**kwargs)
        disp = {'color': (0.0, 0.75, 0.75),
                'opacity': 0.3,
                'shape': 'triangulation',
                'outer_factor': 1.3,
                'inner_factor': 1.}
        for i in range(len(self.elements) - 1):
            self.elements[i].display = deepcopy(disp)
            self.elements[i].display['outer_factor'] = 1.0 * metashellgeometry['r_inner'][i + 1] / metashellgeometry['r_outer'][i]
        self.elements[0].display['inner_factor'] = 0
        self.elements[-1].display = deepcopy(disp)
        self.elements[-1].display['outer_factor'] = 1.5


class MetaShellMirror(optics.FlatStack):

    display = {'shape': 'None'}

    def __init__(self, conf):
        metashellgeometry = conf['metashellgeometry']
        kwargs = {}
        kwargs['position'] = [conf['focallength'], 0, 0]
        kwargs['zoom'] = [20, np.max(metashellgeometry['r_outer'] * 1.1),
                          np.max(metashellgeometry['r_outer'] * 1.1)]
        kwargs['elements'] = [optics.PerfectLens,
                              optics.RadialMirrorScatter]
        kwargs['keywords'] = [{'focallength': conf['focallength']},
                              {'inplanescatter': conf['inplanescatter'],
                               'perpplanescatter': conf['perpplanescatter']}]

        super().__init__(**kwargs)


class MetaShellEfficiency(SimulationSequenceElement):

    def __init__(self):
        data = conf['metashellgeometry']
        eff = conf['metashelleff']
        aeff = np.stack([eff[str(c)].data for c in data['Metashell Serial Number']])
        # factor 100 is to transform cm^2 to mm^2
        ageom = np.pi * (data['r_outer']**2 - data['r_inner']**2)
        self.relative_eff = aeff * 100 / ageom[:, None]
        self.shells = [(i, interp1d(eff['energy'], self.relative_eff[i - 1, :]))
                       for i in data['Metashell Serial Number']]

    def __call__(self, photons):
        for i, func in self.shells:
            ind = photons['shell'] == i
            prob = func(photons['energy'][ind])
            # Risk of extrapolation
            prob = np.clip(prob, 0., 1.)
            photons['probability'][ind] *= prob
        return photons


class MetaShell(Sequence):
    # pass all kwargs to MetaShellMirror.
    # This could be improved (e.g. to allow post_process_steps)
    # but not needed right now.
    def __init__(self, conf):
        super().__init__(elements=[MetaShellMirror(conf),
                                   MetaShellEfficiency(conf)])


class PerfectLensSegment(optics.PerfectLens):
    def __init__(self, conf, **kwargs):
        super().__init__(focallength=conf["focallength"], **kwargs)
        refl =  Table.read(get_pkg_data_filename(f"data/{conf['mirrorcoating']}"),
                           format='ascii.ecsv')
        self.reflectivity_interpolator = RectBivariateSpline(
            refl["energy"],
            [np.deg2rad(float(c)) for c in refl.colnames[1:]],
            np.stack([refl[c] for c in refl.columns[1:]]).T,
        )
