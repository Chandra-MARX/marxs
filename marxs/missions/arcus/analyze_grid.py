'''Analyze a grid of monoenergetic simulations for R and Aeff
'''
import os
import glob
import numpy as np
from astropy.table import QTable, Table
import astropy.units as u
from marxs.analysis.gratings import resolvingpower_from_photonlist_robust as r_from_photons
from marxs.analysis.gratings import effectivearea_from_photonlist
from marxs.base import check_meta_consistent, check_energy_consistent
from marxs.missions.arcus.utils import id_num_offset


def analyze_sim(photons, orders, reference_meta, conf, A_geom):
    '''Get R and Aeff from a single photon event list

    Parameters
    ----------
    photons : `astropy.table.Table`
        Photon event list
    orders : np.array
        list of orders to be analyzed
    reference_meta : dict or None
        If not ``None`` check that the meta information of the photon
        list matches the meta information in ``reference_meta``
    conf : dict
        Arcus configuration (the zero order position for each channel
        is taken from this dict).
    A_geom : quantity
        Geometric area of aperture that was used for photon list.

    Returns
    -------
    res : np.array
        Array of R values for each aperture and order, measured from
        photons that hit a CCD. For signal close to a CCD boundary, the
        detected photon distribution may be artificially narrow.
        In general it is better to use res (because the CCDs don't follow
        the Rowland circle exactly) but it's so close that ``circ_phi``
        can be used for those cases. ``res`` is set to ``np.nan`` if there
        is no signal (e.g. a chip gap).
    Aeff : quantity
        Effective area for each aperture and order, using only photons
        that are not L1 cross-dispersed and detected on a CCD.
    '''
    n_apertures = len(list(id_num_offset.values()))
    photons['aper_id'] = np.digitize(photons['xou'],
                                     bins=list(id_num_offset.values()))

    if reference_meta is not None:
        check_meta_consistent(photons.meta, reference_meta)
    check_energy_consistent(photons)

    chan_name = list(id_num_offset.keys())

    res = np.zeros((n_apertures, len(orders)))
    aeff = np.zeros_like(res) * u.cm**2


    for a in range(1, n_apertures + 1):
        pa = photons[(photons['aper_id'] == a) &
                     (photons['probability'] > 0) &
                     (photons['order_L1'] == 0)]
        plist = [pa[pa['CCD'] >= 0]]
        zeropos = [conf['pos_opt_ax'][chan_name[a - 1]][0]]
        cols = ['proj_x']
        if 'circ_phi' in photons.colnames:
            plist.append(pa[np.isfinite(pa['circ_phi'])])
            zeropos.append(np.arcsin((conf['pos_opt_ax'][chan_name[a - 1]][0]) /
                                      conf['rowland_detector'].r))
            cols = ['proj_x', 'circ_phi']


        res_a, pos_a, std_a = r_from_photons(plist, orders, cols=cols,
                                             zeropositions=zeropos)
        res[a - 1, :] = res_a
        aeff[a - 1, :] = effectivearea_from_photonlist(plist[0], orders,
                                                       len(photons),
                                                       A_geom=A_geom)
        res[aeff == 0] = np.nan
    return res, aeff


def aeffRfromraygrid(inpath, aperture, conf,
                     orders=np.arange(-15, 5),
                     allow_inconsistent_meta=False):
    '''Analyze a grid of simulations for R and Aeff

    Parameters
    ----------
    inpath : string
        Path to the simulations grid
    aperture : `marxs.optics.aperture.BaseAperture`
        Aperture used for the simulation (the geometric opening area is
        taken from this)
    conf : dict
        Arcus configuration (the zero order position for each channel
        is taken from this dict).
    orders : list
        List of integer order numbers to be analyzed
    apertures : list
        List of aperture ids in the simulations
    allow_inconsistent_meta : bool
        If ``False`` (the default) the routine checks that all ray files in
        the grid have compatible meta information (e.g. code version).
        Setting this to ``True`` bypasses that check.

    Returns
    -------
    out : `astropy.table.QTable`
        Table with instrument performance metrics
    '''
    rayfiles = glob.glob(os.path.join(inpath, '*.fits'))
    rayfiles.sort()
    r0 = Table.read(rayfiles[0])
    energies = np.zeros(len(rayfiles)) * u.keV
    # number of apertures is hardcoded
    res = np.zeros((len(rayfiles), 4, len(orders)))
    aeff = np.zeros_like(res) * u.cm**2

    for i, rayfile in enumerate(rayfiles):
        obs = Table.read(rayfile)
        res_i, aeff_i = analyze_sim(obs, orders,
                                    None if allow_inconsistent_meta else r0.meta,
                                    conf, aperture.area)
        res[i, :, :] = res_i
        aeff[i, :, :] = aeff_i
        energies[i] = obs['energy'][0] * obs['energy'].unit

    wave = energies.to(u.Angstrom, equivalencies=u.spectral())
    aeff_4 = aeff.sum(axis=1)
    res_4 = np.ma.masked_invalid(np.ma.average(res,
                                               weights=aeff, axis=1))
    indnon0 = orders != 0
    res_disp = np.ma.average(res_4[:, indnon0],
                             weights=np.ma.masked_equal(aeff_4[:, indnon0], 0),
                             axis=1)
    # Columns get filled when writing to fits
    res_4.fill_value = np.nan
    out = QTable([energies, wave, res,
                 aeff, aeff_4, res_4.filled(np.nan), res_disp],
                 names=['energy', 'wave', 'R',
                        'Aeff', 'Aeff4', 'R4', 'R_disp'])
    out.meta = r0.meta
    for i, o in enumerate(orders):
        out.meta['ORDER_{}'.format(i)] = o
    return out


def orders_from_meta(meta):
    orders = []
    i = 0
    while 'ORDER_{}'.format(i) in meta:
        orders.append(meta['ORDER_{}'.format(i)])
        i += 1
    return np.array(orders)


def csv_per_order(infile, col, outfile):
    '''Rewrite one column in ``aeffRfromraygrid`` to csv file

    Turn one vector-valued (all orders in one cell) column into a
    csv table with one entry per cell.
    '''
    tab = Table.read(infile)
    outtab = Table(tab[col], names=['order_{0}'.format(o) for o
                                    in orders_from_meta(tab.meta)])
    outtab.add_column(tab['wave'], index=0)
    outtab.write(outfile, format='ascii.csv', overwrite=True)
