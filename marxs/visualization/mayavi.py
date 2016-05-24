from __future__ import absolute_import

import numpy as np
import mayavi.mlab

from marxs.math.pluecker import h2e

def plot_rays(data, scalar=None, viewer=None,
              kwargssurface={'colormap': 'Accent', 'line_width': 1, 'opacity': .4}):
    '''Plot lines for simulated rays.

    Parameters
    ----------
    data : np.array of shape(n, N, 3)
        where n is the number of rays, N the number of positions per ray and
        the last dimension is the (x,y,z) of an Eukledian position vector.
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
        raise ValueError('Scalar quantity for each point must have shape ({0},) or ({0}, {1})',format(n, N))

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
