# Licensed under GPL version 3 - see LICENSE.rst
from __future__ import absolute_import
from warnings import warn

import numpy as np

from ..math.pluecker import h2e, e2h
from .utils import format_saved_positions, plot_object_general

from mayavi.mlab import mesh, triangular_mesh


def container(obj, display=None, viewer=None):
    return [plot_object(e, display=None, viewer=viewer) for e in obj.elements]


def plane_with_hole(plane, display, viewer):
    xyz, triangles = plane.triangulate_inner_outer()
    t = triangular_mesh(xyz[:, 0], xyz[:, 1], xyz[:, 2], triangles, color=display['color'])
    return t


def surface(surface, display, viewer, coo1, coo2):
    xyz = surface.parametric_surface(coo1, coo2)
    xyz = h2e(xyz)
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    m = mesh(x, y, z, figure=viewer, color=display['color'])
    return m


def box(box, display, viewer=None):
    corners = np.array([[-1, -1, -1], [-1,+1, -1],
                        [-1, -1,  1], [-1, 1,  1],
                        [ 1, -1, -1], [ 1, 1, -1],
                        [ 1, -1, +1], [ 1, 1, +1]])
    triangles = [(0,2,6), (0,4,6), (0,1,5), (0,4,5), (0,1,3), (0,2,3),
                 (7,3,2), (7,6,2), (7,3,1), (7,5,1), (7,6,4), (7,5,4)]
    corners = np.einsum('ij,...j->...i', box.pos4d, e2h(corners, 1))
    b = triangular_mesh(corners[:,0], corners[:,1], corners[:,2], triangles,
                        color=display['color'])
    return b

def plot_rays(data, scalar=None, viewer=None,
              kwargssurface={'colormap': 'Accent', 'line_width': 1, 'opacity': .4}):
    '''Plot lines for simulated rays.

    Parameters
    ----------
    data : np.array of shape(n, N, 3) or `marxs.simulator.KeepCol` object
        where n is the number of rays, N the number of positions per ray and
        the last dimension is the (x,y,z) of an Eukledian position vector.
        This can also be a ``KeepCol('pos')`` object.
    scalar : None or nd.array of shape (n,) or (n, N)
        This quantity is used to color the rays. If ``None`` all rays will have the same
        color. If it has n elements, each ray will have exactly one color (e.g. color
        according to the energy of the ray), if it has n*N elements, rays will be
        multicolored.
    viewer : ``mayavi.core.scene.Scene instance``.
        If None, the source is not added
        to any figure, and will be added automatically by the modules or filters.
        If False, no figure will be created by modules or filters applied to the
        source: the source can only be used for testing, or numerical algorithms,
        not visualization.
    kwargssurface : dict
        keyword arguments for ``mayavi.mlab.pipeline.surface``
    '''
    if hasattr(data, 'data') and isinstance(data.data, list):
        data = format_saved_positions(data)

    import mayavi.mlab
    # The number of points per line
    N = data.shape[1]
    # number of lines
    n = data.shape[0]

    if scalar is None:
        s = np.zeros(n * N)
    elif scalar.shape == (n, ):
        s = np.repeat(scalar, N)
    elif scalar.shape == (n, N):
        s = scalar
    else:
        raise ValueError('Scalar quantity for each point must have shape ({0},) or ({0}, {1})'.format(n, N))

    x = data[:,:, 0].flatten()
    y = data[:,:, 1].flatten()
    z = data[:,:, 2].flatten()

    a = np.arange(n * N).reshape((-1, N))[:, :-1].flatten()
    b = a + 1
    connections = np.vstack([a,b]).T

    # Create the points
    src = mayavi.mlab.pipeline.scalar_scatter(x, y, z, s, figure=viewer)

    # Connect them
    src.mlab_source.dataset.lines = connections
    src.update()

    # The stripper filter cleans up connected lines
    lines = mayavi.mlab.pipeline.stripper(src, figure=viewer)

    # Finally, display the set of lines
    surface = mayavi.mlab.pipeline.surface(lines, figure=viewer, **kwargssurface)

    return src, lines, surface

plot_registry = {'plane with hole': plane_with_hole,
                 'surface': surface,
                 'box': box,
                 'container': container,
                 }


def plot_object(obj, display=None, viewer=None, **kwargs):
    out = plot_object_general(plot_registry, obj, display, **kwargs)

    if out is not None:
        display = display or obj.display
        # No safety net here like for color converting to a tuple.
        # If the advanced properties are set you are on your own.
        prop = out.module_manager.children[0].actor.property
        for n in prop.trait_names():
            if n in display:
                setattr(prop, n, display[n])
    return out
