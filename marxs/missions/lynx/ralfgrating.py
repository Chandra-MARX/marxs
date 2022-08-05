# Licensed under GPL version 3 - see LICENSE.rst
import numpy as np
from astropy.utils.data import get_pkg_data_filename as gpdf
from astropy.table import Table
from marxs.optics.base import OpticalElement
from marxs.simulator import ParallelCalculated
from marxs.math.utils import h2e
from marxs.missions.mitsnl.catgrating import InterpolateEfficiencyTable as IET

order_selector_Si = IET(gpdf('data/gratings/Si_efficiency.dat',
                             package='marxslynx'))
order_selector_Si.coating = 'None'
order_selector_SiPt = IET(gpdf('data/gratings/SiPt_efficiency.dat',
                               package='marxslynx'))
order_selector_SiPt.coating = 'Pt'

###
###
### To-Do: delete here and use rowland.CircualrMeshGrid instead
class MeshGrid(ParallelCalculated, OpticalElement):
    '''A collection of diffraction gratings on the Rowland torus.

    This class is similar to ``marxs.design.rowland.RectangularGrid`` but
    uses different axes.

    When initialized, it places elements in the space available on the
    Rowland circle, most commonly, this class is used to place grating facets.

    After generation, individual facet positions can be adjusted by hand by
    editing the attributes `elem_pos` or `elem_uncertainty`. See
    `marxs.simulation.Parallel` for details.

    After any of the `elem_pos`, `elem_uncertainty` or
    `uncertainty` is changed, `generate_elements` needs to be
    called to regenerate the facets on the GAS.

    Parameters
    ----------
    rowland : RowlandTorus
    d_element : float
        Size of the edge of a element, which is assumed to be flat and square.
        (``d_element`` can be larger than the actual size of the silicon
        membrane to accommodate a minimum thickness of the surrounding frame.)
    z_range: list of 2 floats
        Minimum and maximum of the x coordinate that is searched for an
        intersection with the torus. A ray can intersect a torus in up to four
        points. ``x_range`` specififes the range for the numerical search for
        the intersection point.
    x_range, y_range: lists of two floats
        limits of the rectangular area where gratings are placed.

    '''

    id_col = 'facet'

    def __init__(self, **kwargs):
        self.radius = kwargs.pop('radius')
        self.x_range = kwargs.pop('x_range')
        self.rowland = kwargs.pop('rowland')
        self.d_element = kwargs.pop('d_element')
        kwargs['pos_spec'] = self.elempos

        super().__init__(**kwargs)

    def elempos(self):

        n_x = int(np.ceil(2 * self.radius[1] / self.d_element[0]))
        x_width = n_x * self.d_element[0]
        x_pos = np.arange(- x_width / 2, self.radius[1], self.d_element[0])

        xpos = []
        ypos = []

        for x in x_pos:
            if np.abs(x) > self.radius[1]:
                # Outermost layer. Center might be outside outer radius
                y = np.array([0])
            else:
                y_outer = np.sqrt(self.radius[1]**2 - x**2)
                if np.abs(x) > self.radius[0]:
                    n_y = int(np.ceil(2 * y_outer / self.d_element[1]))
                    y_width = n_y * self.d_element[1]
                    y = np.arange(- y_width / 2, y_outer, self.d_element[1])
                else:
                    y_inner = np.sqrt(self.radius[0]**2 - x**2)
                    y_mid = 0.5 * (y_inner + y_outer)
                    n_y = int(np.ceil((y_outer - y_inner) / self.d_element[1]))
                    y_width = n_y * self.d_element[1]
                    y = np.arange(y_mid - y_width / 2, y_outer, self.d_element[1])
                    y = np.hstack([-y, y])

            xpos.extend([x] * len(y))
            ypos.extend(y)

        zpos = []
        for x, y in zip(xpos, ypos):
            zpos.append(self.rowland.solve_quartic(y=x, z=y, interval=self.x_range))
        # it's called x/y above, but it's in y/z plane
        return np.vstack([np.array(zpos), np.array(xpos),
                          np.array(ypos), np.ones_like(zpos)]).T
