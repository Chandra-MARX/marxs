# Licensed under GPL version 3 - see LICENSE.rst
from functools import wraps

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.pyplot import get_cmap
from x3d import x3d

from . import utils
from marxs.math import utils as mutils


class Scene(x3d.Scene):
    js_source = 'https://www.x3dom.org/download/x3dom.js'
    css_source = 'https://www.x3dom.org/download/x3dom.css'
    dimension_px = (600, 400)

    # see https://doc.x3dom.org/tutorials/animationInteraction/viewpoint/index.html
    # for how to add buttons for viewpoints

    def _repr_html_(self):

        return(f"""
<html>
  <head>
    <meta http-equiv="X-UA-Compatible" content="IE=edge"/> 
     <script type='text/javascript' src='{self.js_source}'> </script> 
     <link rel='stylesheet' type='text/css' href='{self.css_source}'></link> 
  </head>
  <body>
    <x3d width='{self.dimension_px[0]}px' height='{self.dimension_px[1]}px'> 
      {self.XML()}
    </x3d>
  </body>
</html>
""")


def empty_scene(func):
    @wraps(func)
    def with_scene(*args, **kwargs):
        if not 'scene' in kwargs or kwargs['scene'] is None:
            kwargs['scene'] = Scene(children=[])
        return func(*args, **kwargs)
    return with_scene


@empty_scene
def indexed_triangle_set(xyz, index, display, *, scene):
    scene.children.append(x3d.Shape(appearance=x3d.Appearance(material=x3d.Material(diffuseColor=display['color'])),
                     geometry=x3d.IndexedTriangleSet(coord=x3d.Coordinate(point=[tuple(p) for p in xyz]),
                                                     index=[int(i) for i in index.reshape(-1, 3).flatten()],
                                                     solid=False, colorPerVertex=False)))

#@empty_scene
#def surface(surface, display, *, scene):
#    # better used IndexedFaceSet ?
#    xyz = surface.geometry.parametric_surface(display.get('coo1', None),
#                                              display.get('coo2', None),
#                                              display)
#    xyz, index = utils.triangulate_parametricsurface(xyz)
#    indexed_triangle_set(xyz, index, display, scene)
#    return scene

@empty_scene
def surface(surface, display, *, scene):
    xyz = surface.geometry.parametric_surface(display.get('coo1', None),
                                              display.get('coo2', None),
                                              display)
    xyz = mutils.h2e(xyz)
    # number of faces. "-1" because last row just closes last face, but does not start a new one.
    n = xyz.shape[0] - 1
    # Each face has 4 vertices, 0 1 2 3, then 2 3 4 5, so need to step by 2 for each new face
    # [0, 2, 3, 1] is simply the right order to go around the polygon
    index = np.vstack([np.arange(i, i + 2 * n, 2, dtype=int) for i in [0, 2, 3, 1]] + 
                      # Add -1 at the end of each row to mark end of each face
                      [-1 * np.ones(n, dtype=int)]).T.flatten()
    scene.children.append(x3d.Shape(appearance=x3d.Appearance(material=x3d.Material(diffuseColor=display['color'])),
                                    geometry=x3d.IndexedFaceSet(coord=
                                        x3d.Coordinate(point=[tuple(p) for p in xyz.reshape((-1, 3))]),
                                        coordIndex=[int(i) for i in index],
                                        solid=False, colorPerVertex=False)))


@empty_scene
def triangulation(obj, display, *, scene):
    '''Plot a plane with an inner hole such as an aperture.'''

    xyz, index = obj.geometry.triangulate(display)
    indexed_triangle_set(xyz, index, display, scene=scene)
    return scene


@empty_scene
def box(obj, display, *, scene):
    corners = utils.halfbox_corners(obj, display)
    shape = x3d.Shape(appearance=x3d.Appearance(material=x3d.Material(diffuseColor=display['color'])),
                     geometry=x3d.IndexedFaceSet(coord=x3d.Coordinate(point=[tuple(p) for p in corners]),
                                                 coordIndex=[0, 2, 3, 1, -1, 
                                                             4, 6, 7, 5, -1,
                                                             0, 4, 6, 2, -1,
                                                             1, 5, 7, 3, -1,
                                                             0, 4, 5, 1, -1,
                                                             2, 6, 7, 3, -1],
                                                  solid=False, colorPerVertex=False))
    scene.children.append(shape)

@empty_scene
def container(obj, display=None, *, scene=None):
    '''Recursively plot objects containted in a container.'''
    for e in obj.elements:
        plot_object(e, display=None, scene=scene)
    return scene


@empty_scene
def plot_object(obj, display=None, *, scene=None):

    utils.plot_object_general(plot_registry, obj, display, scene=scene)
    return scene


@empty_scene
def plot_rays(data, scalar=None, *, scene=None, cmap=get_cmap('jet')):
    '''Plot lines for simulated rays.

    Parameters
    ----------
    data : np.array of shape(n, N, 3)
        where n is the number of rays, N the number of positions per ray and
        the last dimension is the (x,y,z) of an Eukledian position vector.
    scalar : None or nd.array of shape (n,) or (n, N)
        CURRENTLY COLOR PER LINE BUT COULD USER colorPerVertex for multi-colored lines
    scene :
    kwargssurface : dict
        keyword arguments for ``mayavi.mlab.pipeline.surface``

    Returns
    -------
    out : mayavi ojects
        This just passes through the information returned from the mayavi calls.

    '''
    # The number of points per line
    N = data.shape[1]
    # number of lines
    n = data.shape[0]

    if scalar is None:
        color = [(0, 0, 0) for n in range(n)]
    elif scalar.shape == (n, ):
        from matplotlib.colors import Colormap, Normalize
        scalar = Normalize()(scalar)
        color = cmap(scalar)
        # color is RGBA, but I have not figured out the alpha in X3D, so just drop that
        # and use RGB
        color = [tuple(c[:3]) for c in color]
    else:
        raise ValueError('Scalar quantity for each point must be scalar or have shape({0},) '.format(n, N))\

    # from 0, .. N for each line
    ind = np.arange(N * n).reshape(-1, N)
    # add a -1 at the end of each line to indicate that a new line is starting
    index = np.hstack([ind, -1 * np.ones((n, 1), dtype=int)])

    lines = x3d.Shape(geometry=x3d.IndexedLineSet(colorPerVertex=False,
                                                  coordIndex=[int(i) for i in index.flatten()],
                                                  coord=x3d.Coordinate(point=[tuple(p) for p in data.reshape(-1, 3)]),
                                                  color=x3d.Color(color=color),
                                                  ))
    scene.children.append(lines)
    return scene


plot_registry = {'triangulation': triangulation,
                 'box': box,
                 'container': container,
                 'surface': surface,
                 }