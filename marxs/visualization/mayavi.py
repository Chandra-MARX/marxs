# Licensed under GPL version 3 - see LICENSE.rst
'''`Mayavi <http://docs.enthought.com/mayavi/mayavi/>`__ plotting backend

Mayavi is a python package for interactive 3-D displays that uses VTK underneath.
Fuctions in this module display rays or objects in 3D using mayavi. Each of them
requires a ``mayavi.core.scene.Scene`` instance as input and returns a mayavi object
(or a list of those), e.g. a ``mayavi.visual.Box`` instance.

The plotting routines attempt to find all valid OpenGL properties by name in the
``display`` dictionaries and apply those to the plotted object.
'''
from __future__ import absolute_import

import numpy as np
from astropy.utils.decorators import format_doc

from ..math import utils as mutils
from . import utils

# The following import fails on headless servers (Travis, readthedocs).
# from mayavi import mlab
# Thus, I've moved the import statement into the individual functions, so that
# this module can still be imported when Travis or readthedocs build the documentation.

doc_plot='''
    {__doc__}

    Parameters
    ----------
    obj : `marxs.base.MarxsElement`
        The element that should be plotted.
    display : dict of None
        Dictionary with display settings.
    viewer : ``mayavi.core.scene.Scene instance``.
        If None, the source is not added
        to any figure, and will be added automatically by the modules or filters.
        If False, no figure will be created by modules or filters applied to the
        source: the source can only be used for testing, or numerical algorithms,
        not visualization.

    Parameters
    ----------
    out : mayavi object
        Return the result of a mayavi plotting method.
'''

@format_doc(doc_plot)
def container(obj, display=None, viewer=None):
    '''Recursively plot objects containted in a container.'''
    return [plot_object(e, display=None, viewer=viewer) for e in obj.elements]

@format_doc(doc_plot)
def triangulation(obj, display, viewer=None):
    '''Plot a plane with an inner hole such as an aperture.'''
    from mayavi import mlab

    xyz, triangles = obj.triangulate()
    t = mlab.triangular_mesh(xyz[:, 0], xyz[:, 1], xyz[:, 2], triangles, color=display['color'])
    return t

@format_doc(doc_plot)
def surface(surface, display, viewer=None):
    '''Plot a parametric surface.

    The parameter boundaries are taken from the ``coo1`` and ``coo2`` in the
    display dictionary. The plotting routine is generic. It calls the
    ``parametric_surface()`` method of the object that is plotted; see there
    for a detailted description of parameters.
    '''
    from mayavi import mlab

    xyz = surface.parametric_surface(display.get('coo1', [-1, 1]),
                                     display.get('coo2', [-1, 1]))
    xyz = mutils.h2e(xyz)
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    m = mlab.mesh(x, y, z, figure=viewer, color=display['color'])
    return m

@format_doc(doc_plot)
def box(obj, display, viewer=None):
    '''Plot a rectangular box for an object.'''
    from mayavi import mlab

    corners = np.array([[-1, -1, -1], [-1,+1, -1],
                        [-1, -1,  1], [-1, 1,  1],
                        [ 1, -1, -1], [ 1, 1, -1],
                        [ 1, -1, +1], [ 1, 1, +1]])
    triangles = [(0,2,6), (0,4,6), (0,1,5), (0,4,5), (0,1,3), (0,2,3),
                 (7,3,2), (7,6,2), (7,3,1), (7,5,1), (7,6,4), (7,5,4)]
    corners = np.einsum('ij,...j->...i', obj.pos4d, mutils.e2h(corners, 1))
    b = mlab.triangular_mesh(corners[:,0], corners[:,1], corners[:,2],
                             triangles, color=display['color'])
    return b

@format_doc(doc_plot)
def cylinder(obj, display, viewer=None):
    '''Plot a rectangular box for an object.'''
    from mayavi import mlab

    x0 = obj.geometry('center') - obj.geometry('v_x')
    x1 = obj.geometry('center') + obj.geometry('v_x')
    c = mlab.plot3d([x0[0], x1[0]], [x0[1], x1[1]], [x0[2], x1[2]],
                    color=display['color'],
                    tube_radius=np.linalg.norm(obj.geometry('v_y')),
                    tube_sides=display.get('tube_sides', 20))
    return c

def plot_rays(data, scalar=None, viewer=None,
              kwargssurface={'colormap': 'Accent', 'line_width': 1, 'opacity': .4}):
    '''Plot lines for simulated rays.

    Parameters
    ----------
    data : np.array of shape(n, N, 3)
        where n is the number of rays, N the number of positions per ray and
        the last dimension is the (x,y,z) of an Eukledian position vector.
    scalar : None or nd.array of shape (n,) or (n, N)
        This quantity is used to color the rays. If ``None`` all rays will have
        the same color. If it has n elements, each ray will have exactly one
        color (e.g. color according to the energy of the ray), if it has n*N
        elements, rays will be multicolored.
    viewer : ``mayavi.core.scene.Scene instance``.
       If None, the source is not added to any figure, and will be added
        automatically by the modules or filters.  If False, no figure will be
        created by modules or filters applied to the source: the source can
        only be used for testing, or numerical algorithms, not visualization.
    kwargssurface : dict
        keyword arguments for ``mayavi.mlab.pipeline.surface``

    Returns
    -------
    out : mayavi ojects
        This just passes through the information returned from the mayavi calls.

    '''
    from mayavi import mlab

    # The number of points per line
    N = data.shape[1]
    # number of lines
    n = data.shape[0]

    if scalar is None:
        s = np.zeros(n * N)
    elif scalar.shape == (n, ):
        s = np.repeat(scalar, N)
    elif scalar.shape == (n, N):
        s = scalar.flatten()
    else:
        raise ValueError('Scalar quantity for each point must have shape ({0},) or ({0}, {1})'.format(n, N))

    x = data[:,:, 0].flatten()
    y = data[:,:, 1].flatten()
    z = data[:,:, 2].flatten()

    a = np.arange(n * N).reshape((-1, N))[:, :-1].flatten()
    b = a + 1
    connections = np.vstack([a,b]).T

    # Create the points
    src = mlab.pipeline.scalar_scatter(x, y, z, s, figure=viewer)

    # Connect them
    src.mlab_source.dataset.lines = connections
    src.update()

    # The stripper filter cleans up connected lines
    lines = mlab.pipeline.stripper(src, figure=viewer)

    # Finally, display the set of lines
    surface = mlab.pipeline.surface(lines, figure=viewer, **kwargssurface)

    return src, lines, surface

plot_registry = {'triangulation': triangulation,
                 'surface': surface,
                 'box': box,
                 'container': container,
                 'cylinder': cylinder,
                 }


def plot_object(obj, display=None, viewer=None, **kwargs):
    '''Plot any marxs object with using Mayavi as a backend.

    This method will inspect the object that is passed in and select the
    correct plotting method for its shape. The object is added to the
    mayavi scene specified in the ``viewer``.


    Parameters
    ----------
    obj : `marxs.base.MarxsElement`
        The element that should be plotted.
    display : dict of None
        Dictionary with display settings. If this is ``None``, ``obj.display``
        is used. If that is also ``None`` then the objects is skipped.
    viewer : ``mayavi.core.scene.Scene``
        If None, the source is not added to any figure, and will be added
        automatically by the modules or filters.  If False, no figure will be
        created by modules or filters applied to the source: the source can
        only be used for testing, or numerical algorithms, not visualization.
    kwargs
        All other parameters will be passed on to the individual plotting
        method.

    Parameters
    ----------
    out : mayavi object
        Return the result of a mayavi plotting method.

    '''
    kwargs['viewer'] = viewer
    out = utils.plot_object_general(plot_registry, obj, display, **kwargs)

    if (out is not None) and not isinstance(out, list):
        # containers return list, but properties are already set there
        display = display or obj.display
        # No safety net here like for color converting to a tuple.
        # If the advanced properties are set you are on your own.
        prop = out.module_manager.children[0].actor.property
        for n in prop.trait_names():
            if n in display:
                setattr(prop, n, display[n])
    return out
