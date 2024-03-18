# Licensed under GPL version 3 - see LICENSE.rst
'''`X3D <https://www.web3d.org/x3d/what-x3d>`__ plotting backend

X3D is an open standard for interactive 3D models on the Web.
X3D models are stored in XML files and can be viewed in the
Jupyter notebook or on the web with different js libraries
without the installation of a plug-in.

This module implements the `~marxs.visualization.x3d.Scene`
class, which can represent itself as a complete HTML page to be
rendered directly or it can return the XML format to be saved
and used in other contexts.

Each plotting routine requires a `~marxs.visualization.x3d.Scene`
as input and objects are added to it (if no scene is passed in,
an empty one is generated.)

Under the hood, the XML is constructed using the
`x3d <https://pypi.org/project/x3d/>`__ Python package.
'''
from functools import wraps
from warnings import warn
import os
import urllib.request
import tempfile
from shutil import make_archive
import xml.etree.ElementTree as ET

import numpy as np
from astropy.utils.decorators import format_doc
from matplotlib.colors import Normalize
from matplotlib.pyplot import get_cmap
from x3d import x3d

from . import utils
from . import conf
from marxs.math import utils as mutils

__all__ = ['Scene',
           'empty_scene',
           'indexed_triangle_set',
           'surface',
           'triangulation',
           'box',
           'container',
           'plot_object',
           'plot_rays',
           'plot_registry',
           ]


doc_plot = '''
    {__doc__}

    Parameters
    ----------
    obj : `marxs.base.MarxsElement`
        The element that should be plotted.
    display : dict of None
        Dictionary with display settings.
    scene : `marxs.visualization.x3d.Scene` object
        A scene that this object is added to.
        If `None`, a new scene will be created.

    Returns
    -------
    scene : `marxs.visualization.x3d.Scene` object
        Scene with object added.
'''

class Scene(x3d.Scene):
    '''X3D Scene with added _repr_html_ for notebook output'''
    js_source = 'https://www.x3dom.org/download/x3dom.js'
    css_source = 'https://www.x3dom.org/download/x3dom.css'
    dimension_px = (600, 400)

    # see https://doc.x3dom.org/tutorials/animationInteraction/viewpoint/index.html
    # for how to add buttons for viewpoints

    def _repr_html_(self):
        root = ET.fromstring(self.XML())
        html = f"""
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
    """
        for vp in root.findall('Viewpoint'):
            description = vp.get('description')
            html += f'<button onclick="document.querySelector(\'Viewpoint[description=\\\'{description}\\\']\').setAttribute(\'set_bind\',\'true\');">{description}</button>\n'
        html +="""
  </body>
</html>
"""
        return html

    def write_html_archive(self, base_name: str, format: str, *args, **kwargs) -> None:
        """Write to an HTML archive with local copies of js and css requirements.

        This zip file can be shared with others and opened in a web browser.
        In particular, it can be used for journal submission for journals
        that accept interactive js-based content such as ApJ or AJ.

        Parameters
        ----------
        base_name : str
            Name of the file to create, including the path,
            minus any format-specific extension.

        format : {"zip", "tar", "gztar", "bztar", "xztar"}
            The archive format, see `shutil.make_archive` for details.

        *args : tuple
            Other arguments are passed to `shutil.make_archive`.

        **kwargs : dict, optional
            Other keyword arguments are passed to `shutil.make_archive`
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            for f in (self.js_source, self.css_source):
                urllib.request.urlretrieve(
                    f, os.path.join(tmpdirname, os.path.basename(f))
                )
            try:
                old_js = self.js_source
                old_css = self.css_source
                self.js_source = os.path.basename(self.js_source)
                self.css_source = os.path.basename(self.css_source)
                with open(os.path.join(tmpdirname, "figure.html"), "w") as f:
                    f.write(self._repr_html_())
            finally:
                self.js_source = old_js
                self.css_source = old_css
            make_archive(base_name, format, root_dir=tmpdirname, *args, **kwargs)


def empty_scene(func):
    @wraps(func)
    def with_scene(*args, **kwargs):
        if 'scene' not in kwargs or kwargs['scene'] is None:
            kwargs['scene'] = Scene(children=[])
        return func(*args, **kwargs)
    return with_scene


def _diffuse_material(display):
    return x3d.Material(diffuseColor=display['color'],
                        transparency=1 - display.get('opacity', 1.))


def _format_points(xyz):
    return [tuple(np.round(p, conf.xyz_precision)) for p in xyz]


@empty_scene
def indexed_triangle_set(xyz, index, display, *, scene):
    '''Plot a set of triangles.

    Parameters
    ----------
    xyz : np.array
        Euclidean coordinates of nodes
    index : np.array
        index of nodes for each triangle
    display : dict of None
        Dictionary with display settings.
    scene : `marxs.visualization.x3d.Scene` object
        A scene that this object is added to.
        If `None`, a new scene will be created.

    Returns
    -------
    scene : `marxs.visualization.x3d.Scene` object
        Scene with object added.
    '''
    scene.children.append(x3d.Shape(appearance=x3d.Appearance(material=_diffuse_material(display)),
                     geometry=x3d.IndexedTriangleSet(coord=x3d.Coordinate(point=_format_points(xyz)),
                                                     index=[int(i) for i in index.reshape(-1, 3).flatten()],
                                                     solid=False, colorPerVertex=False)))


@empty_scene
@format_doc(doc_plot)
def surface(obj, display, *, scene):
    '''Plot a parametric surface.

    The parameter boundaries are taken from the ``coo1`` and ``coo2`` in the
    display dictionary. The plotting routine is generic. It calls the
    ``parametric_surface()`` method of the object that is plotted; see there
    for a detailed description of parameters.
    '''
    xyz = obj.geometry.parametric_surface(display.get('coo1', None),
                                          display.get('coo2', None),
                                          display)
    xyz = mutils.h2e(xyz)
    # number of faces. "-1" because last row just closes last face, but does not start a new one.
    # Each face has 4 vertices
    # [0, 2, 3, 1] is simply the right order to go around the polygon
    ind = np.arange(xyz.shape[0] * xyz.shape[1], dtype=int).reshape(xyz.shape[0], xyz.shape[1])
    coordIndex = np.stack([ind[:-1, :-1].flatten(),
                           ind[:-1, 1:].flatten(),
                           ind[1:, 1:].flatten(),
                           ind[1:, :-1].flatten(),
                            - np.ones((ind.shape[0] - 1) * (ind.shape[1] - 1),
                                      dtype=int)]).T
    coordIndex = coordIndex.flatten()
    scene.children.append(x3d.Shape(appearance=x3d.Appearance(material=_diffuse_material(display)),
                                    geometry=x3d.IndexedFaceSet(coord=
                                        x3d.Coordinate(point=_format_points(xyz.reshape((-1, 3)))),
                                        coordIndex=list(coordIndex),
                                        solid=False, colorPerVertex=False)))


@empty_scene
@format_doc(doc_plot)
def triangulation(obj, display, *, scene):
    '''Plot a plane, e.g. an aperture with an inner hole.'''
    xyz, index = obj.geometry.triangulate(display)
    indexed_triangle_set(xyz, index, display, scene=scene)
    return scene


@empty_scene
@format_doc(doc_plot)
def box(obj, display, *, scene):
    '''Plot a rectangular box for an object.

    By default, the box extends in x,y, and z direction. The display keyword
    "box-half" can be used to show only one half of the box, e.g. "+x" would
    show  the full extend in y and z direction, but only the lower half in
    x direction (such that rays coming from the +x direction are
    visible up to the interaction point).
    Use this for elements such as mirrors or detectors where
    photon interaction happens on the surface, not in the substrate.
    '''
    corners = utils.halfbox_corners(obj, display)
    shape = x3d.Shape(
        appearance=x3d.Appearance(material=_diffuse_material(display)),
        geometry=x3d.IndexedFaceSet(
            coord=x3d.Coordinate(point=_format_points(corners)),
            coordIndex=[
                0,
                2,
                3,
                1,
                -1,
                4,
                6,
                7,
                5,
                -1,
                0,
                4,
                6,
                2,
                -1,
                1,
                5,
                7,
                3,
                -1,
                0,
                4,
                5,
                1,
                -1,
                2,
                6,
                7,
                3,
                -1,
            ],
            solid=False,
            colorPerVertex=False,
        ),
    )
    scene.children.append(shape)

@empty_scene
@format_doc(doc_plot)
def container(obj, display=None, *, scene=None):
    '''Recursively plot objects contained in a container.'''
    for e in obj.elements:
        plot_object(e, display=None, scene=scene)
    return scene


@empty_scene
def plot_object(obj, display=None, *, scene=None, **kwargs):
    '''Plot any marxs object with using X3D as a backend.

    This method will inspect the object that is passed in and select the
    correct plotting method for its shape. The object is added to the
    x3d scene specified in the ``scene``.


    Parameters
    ----------
    obj : `marxs.base.MarxsElement`
        The element that should be plotted.
    display : dict of None
        Dictionary with display settings. If this is ``None``, ``obj.display``
        is used. If that is also ``None`` then the objects is skipped.
    scene : `marxs.visualization.x3d.Scene` object
        A scene that this object is added to.
        If `None`, a new scene will be created.
    kwargs
        All other parameters will be passed on to the individual plotting
        method.

    Returns
    -------
    scene : `marxs.visualization.x3d.Scene` object
        Scene with object added.
    '''
    utils.plot_object_general(plot_registry, obj, display, scene=scene, **kwargs)
    return scene


@empty_scene
def plot_rays(
    data, scalar=None, *, scene=None, cmap=get_cmap("viridis"), normalize_kwargs={}
):
    """Plot lines for simulated rays.

    Parameters
    ----------
    data : np.array of shape(n, N, 3)
        where n is the number of rays, N the number of positions per ray and
        the last dimension is the (x,y,z) of an Euclidean position vector.
    scalar : None or nd.array of shape (n,) or (n, N)
    scene : `marxs.visualization.x3d.Scene` object
        A scene that rays are added to.
        If `None`, a new scene will be created.
    normalize_kwargs : dict
        Keyword arguments for the normalization of the scalar quantity
        see `matplotlib.colors.Normalize` for accepted keywords.

    Returns
    -------
    scene : `marxs.visualization.x3d.Scene` object
        Scene with object added.
    """
    # The number of points per line
    N = data.shape[1]
    # number of lines
    n = data.shape[0]

    if scalar is None:
        # Color all rays the same
        scalar = np.ones(n)
    elif scalar.shape == (n, ):
        pass
    elif scalar.shape == (n, N):
        warn('Color Per Vertex does not yet work in the X3D library used. Using only color for first node in each ray.')
        scalar = scalar[0, :]
    else:
        raise ValueError('Scalar quantity for each point must have shape ({0},) or ({0}, {1})'.format(n, N))

    scalar = Normalize(**normalize_kwargs)(scalar)
    scalarset = set(scalar)
    # If scalar is a float quantity it makes smaller X3D file to bin and have e.g. no more than 100 colors.
    # Not hard to implement, but punt for now since I don't know if that optimization is needed in
    # practice.
    for s in scalarset:
        color = cmap(s)
        color = tuple(np.round(color, conf.color_precision))
        # color is RGBA, but I have not figured out the alpha in X3D, so just drop that
        # and use RGB
        # Also, round to two digits to reduce X3D filesize
        # color = x3d.Color(tuple(np.round(color[:3], 2)))
        ind = scalar == s

        lines = x3d.Shape(
            appearance=x3d.Appearance(material=x3d.Material(emissiveColor=color[:3])),
            geometry=x3d.LineSet(vertexCount=[N] * ind.sum(),
                                 # Rounding to two post-comma digits (i.e. 0.1 mm in MARXS default units)
                                 # to keep file size down
                                coord=x3d.Coordinate(point=_format_points(data[ind, :].reshape(-1, 3))),
                                )
        )
        scene.children.append(lines)
    return scene



plot_registry = {'triangulation': triangulation,
                 'box': box,
                 'container': container,
                 'surface': surface,
                 }