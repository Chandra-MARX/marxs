# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from astropy.table import Table

from ..optics import FlatOpticalElement
from ..math.utils import h2e


__all__ = ['ProjectOntoPlane', 'facet_table']


class ProjectOntoPlane(FlatOpticalElement):
    '''Project photon positions onto a plane.

    *Projection* as used in this class differs from propagating the photons
    until they intersect with a plane. Instead, the current photon positions
    are projected parallel to the normal of the plane.

    Note that the default output columns are called proj_x and _y independent
    of the actual orientation of the plane.

    Examples
    --------
    Project some points onto a plane. If no orientation is given to the
    ``ProjectionOntoPlane`` object, the default is to place it in the yz-plane.

    >>> from astropy.table import Table
    >>> import numpy as np
    >>> from marxs.analysis.coords import ProjectOntoPlane
    >>> photons = Table()
    >>> photons['pos'] = np.array([[3, 3, 3, 1], [12, -1, 0, 1]])
    >>> yzplane = ProjectOntoPlane()
    >>> photons = yzplane(photons)
    >>> photons['proj_x'].data
    array([ 3., -1.])
    >>> photons['proj_y'].data
    array([3., 0.])
    '''

    loc_coos_name = ['proj_x', 'proj_y']
    '''name for output columns of the projected position in plane coordinates.'''

    display = {'shape': 'None'}

    def __call__(self, photons):
        vec_center_inter = - h2e(self.geometry['center']) + h2e(photons['pos'])
        photons[self.loc_coos_name[0]] = np.dot(vec_center_inter, h2e(self.geometry['e_y']))
        photons[self.loc_coos_name[1]] = np.dot(vec_center_inter, h2e(self.geometry['e_z']))
        return photons


def facet_table(container, project_plane=ProjectOntoPlane()):
    '''Get table of facet properties

    Parameters
    ----------
    container : `marxs.simulator.Parallel` instance
        This can be some kind of container, typically a grating array
        on the Rowland torus.
    project_plane : `marxs.analysis.ProjectOntoPlane`
        radius and angle are calculated as projected onto this plane

    Returns
    -------
    facettab : `astropy.table.Table`
        Table with facet properties.
        All output columns are prefaced with "facet_" to avoid a clash with columns likely
        to be found in photon tables. That enables easy merging of facet tables with photon tables
        to add the properties of a facet that the photon passed through into the photon table
        like so:

            >>> from astropy import table
            >>> table.join(photons, facettab, join_type='left')  # doctest: +SKIP
    '''
    project_plane.loc_coos_name = ['facet_projx', 'facet_projy']

    facetpos = h2e(np.stack(container.elem_pos)[:, :, 3])
    tab = Table({'pos': np.stack(container.elem_pos)[:, :, 3]})
    tab = project_plane(tab)

    tab['facet_ang'] = np.arctan2(tab['facet_projy'], tab['facet_projx'])
    # projx and proy are measured from center of plane, not from origin of cooridnate system!
    tab['facet_rad'] = np.sqrt(tab['facet_projy']**2 + tab['facet_projx']**2)
    tab['facet'] = [e.id_num for e in container.elements]
    for i, s in enumerate('xyz'):
        tab[f'facet_{s}'] = facetpos[:, i]
    tab.remove_column('pos')

    return tab
