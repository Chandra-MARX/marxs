import numpy as np
import transforms3d
import astropy.table
from astropy.stats import sigma_clipped_stats

from marxs.source import ConstantPointSource, FixedPointing
from marxs.design import RowlandTorus, find_radius_of_photon_shell, GratingArrayStructure
from marxs.optics import MarxMirror, FlatGrating, uniform_efficiency_factory, FlatDetector, EfficiencyFile, constant_order_factory
from marxs.design.rowland import design_tilted_torus
from marxs.math.pluecker import h2e
from marxs.analysis import find_best_detector_position, measure_FWHM

def fwhm_per_order(gas, photons, orders=np.arange(-11,-1)):
    '''Calculate FWHM in every order.

    As a side effect, the function that selects the grating orders for diffraction in ``gas``
    will be changed. Pass a deep copy of the GAS if this could affect consecutive computations.

    Parameters
    ----------
    gas : `marxs.design.rowland.GratingArrayStructure`
    photons : `astropy.table.Table`
        Photon list to be processed for every order
    orders : np.array of type int
        Order numbers
    '''
    res = np.zeros_like(orders, dtype=float)
    fwhm = np.zeros_like(orders, dtype=float)
    det_x = np.zeros_like(orders, dtype=float)

    for i, order in enumerate(orders):
        gratingeff = constant_order_factory(order)
        gas.elem_args['order_selector'] = gratingeff
        for facet in gas.elements:
            facet.order_selector = gratingeff

        pg = photons.copy()
        pg = gas.process_photons(pg)
        pg = pg[pg['order'] == order]  # Remove photons that slip between the gratings
        xbest = find_best_detector_position(pg, objective_func=measure_FWHM)
        fwhm[i] = xbest.fun
        det_x[i] = xbest.x
        meanpos, medianpos, stdpos = sigma_clipped_stats(pg['det_y'])
        res[i] = np.abs(meanpos / xbest.fun)
    return fwhm, det_x, res

def run_res_sim(energy, blaze=0):
    mysource = ConstantPointSource((30., 30.), energy=energy, flux=1.)
    mypointing = FixedPointing(coords=(30, 30.))
    marxm = MarxMirror('../marxs/optics/hrma.par', position=np.array([0., 0,0]))

    blazeang = np.deg2rad(blaze)
    blazemat = transforms3d.axangles.axangle2mat(np.array([0,1,0]), np.deg2rad(blazeang))
    R, r, pos4d = design_tilted_torus(9e3, np.deg2rad(blazeang), np.deg2rad(2*blazeang))
    mytorus = RowlandTorus(R, r, pos4d=pos4d)

    photons = mysource.generate_photons(100000)
    photons = mypointing.process_photons(photons)
    photons = marxm.process_photons(photons)
    photons = photons[(photons['probability'] > 0) & (photons['mirror_shell'] == 0)]

    # design the gratings
    radius0 = find_radius_of_photon_shell(photons, 0, 9e3)
    mytorus = RowlandTorus(9e3/2, 9e3/2)
    mygas = GratingArrayStructure(mytorus, d_facet=15., x_range=[5e3,1e4], radius=[538., 550.], elem_class=FlatGrating, elem_args={'zoom': 10, 'orientation': blazemat, 'd':0.0002, 'order_selector': None})


    fwhm, det_x, res = fwhm_per_order(mygas, photons, np.arange(-10, 0))

    # design a sub-aperture GAS
    mygas.elements = [e for e in mygas.elements if abs(e.pos4d[1, 3]) < 100 ]
    fwhms, det_xs, ress = fwhm_per_order(mygas, photons, np.arange(-10, 0))
    return np.vstack([fwhm, det_x, res, fwhms, det_xs, ress])

#catfile = '/melkor/d1/guenther/marx/xraysurveyor/sim_input/Si-ox_p200_th15_dc02_d6110.dat'
#gratingeff = EfficiencyFile(catfile, orders=np.arange(2, -13, -1))

# from IPython.parallel import Client
# from grating_resolution import run_res_sim
# rc = Client()
# dview = rc[:]
# energies = np.arange(0.5, 5., .5)
# parallel_result = dview.map_sync(run_res_sim, energies)
# res = np.dstack(parallel_result)
