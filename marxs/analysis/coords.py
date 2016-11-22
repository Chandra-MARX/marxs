import numpy as np

from ..optics import FlatOpticalElement
from ..math.pluecker import h2e

class ProjectOntoPlane(FlatOpticalElement):
    '''Project photon positions onto a plane.

    *Projection* as used in this class differs from propagating the photons
    until they intersect with a plane. Instead, the current photon positions
    are projected parallel to the normal of the plane.

    Note that the default output columns are called proj_x and _y independent
    of the actual orientation of the plane.

    Example
    -------
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
    array([ 3., 0.])
    '''

    loc_coos_name = ['proj_x', 'proj_y']
    '''name for output columns of the projected position in plane coordinates.'''

    def process_photons(self, photons):
        vec_center_inter = - h2e(self.geometry('center')) + h2e(photons['pos'])
        photons[self.loc_coos_name[0]] = np.dot(vec_center_inter, h2e(self.geometry('e_y')))
        photons[self.loc_coos_name[1]] = np.dot(vec_center_inter, h2e(self.geometry('e_z')))
        return photons
