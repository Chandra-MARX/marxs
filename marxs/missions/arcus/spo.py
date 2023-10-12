# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np


__all__ = ['rmid_spo', 'zoom_to_cover_all_spos']

def rmid_spo(conf):
    '''Looking at SPO positions, find midpoint of all SPOs.

    To simplify this code it only considers the mid-points of the SPOs, not
    their rotation angle. Since the rectangles are rotated a bit, e.g.
    the aperture needs to be a bit bigger than the area covered by SPOs.
    Also, if elements are above or below the SPOs, the beam may converge
    or come from the side. `azscale` and `rscale` can be set to adjust
    the outer corners a bit.

    Parameters
    ----------
    conf : dict
        Configuration dictionary with entry 'spo_geom' and 'spo_pos4d'

    Returns
    -------
    zoom : float
        middle of the SPO array in y direction
    '''
    spopos = np.array(conf['spo_pos4d'])
    return 0.5 * (spopos[:, 1, 3].max() + spopos[:, 1, 3].min())


def zoom_to_cover_all_spos(conf, azscale=.8, rscale=1.1):
    '''Looking at SPO positions, find zoom of rectangle that covers it.

    To simplify this code it only considers the mid-points of the SPOs, not
    their rotation angle. Since the rectangles are rotated a bit, e.g.
    the aperture needs to be a bit bigger than the area covered by SPOs.
    Also, if elements are above or below the SPOs, the beam may converge
    or come from the side. `azscale` and `rscale` can be set to adjust
    the outer corners a bit.

    Parameters
    ----------
    conf : dict
        Configuration dictionary with entry 'spo_geom' and 'spo_pos4d'
    azscale : float
        The outer edge is set to the center of the outermost SPO, plus
        `azscale` times the dimension in azimuth (global x coordinate)
    rscale : float
        The outer edge is set to the center of the outermost SPO, plus
        `rscale` times the dimension in radius (global y coordinate)

    Returns
    -------
    zoom : np.array of shape (3,)
        Zoom array as input to make pos4d
    '''
    spopos = np.array(conf['spo_pos4d'])
    azmid = 0.5 * (spopos[:, 0, 3].max() + spopos[:, 0, 3].min())
    rmid = rmid_spo(conf)
    delta_az = conf['spo_geom']['azwidth']
    delta_r = conf['spo_geom']['outer_radius'] - conf['spo_geom']['inner_radius']
    return np.array([1,
                     spopos[:, 0, 3].max() - azmid + azscale * delta_az.max(),
                     spopos[:, 1, 3].max() - rmid + rscale * delta_r.max()])