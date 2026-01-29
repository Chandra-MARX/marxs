# Licensed under GPL version 3 - see LICENSE.rst
'''Plot routines for json output to be loaded into `three.js <threejs.org>`__.

Each routine returns a json dictionary. These json dictionaries can be collected in
a list and  written to a file with `~marxs.visualization.threesjson.write`, which
checks the formatting and adds metadata.

Just as `~marxs.visualization.threejs` this backend provides input for a webpage that
will then use the `three.js <threejs.org>`_ library to render the 3D model. The main
difference is that `~marxs.visualization.threejs` outputs plain text for each object,
while the json written by this module is much smaller if the model contains many copies
of the same object (e.g. hundreds of diffraction gratings). Also, the json data can be
updated with no changes necessary to the html page.

MARXS includes ``loader.html`` as an example how ``MARXSloader.js`` (included in MARXS
as well) can be used to load the json file and construct the three.js objects.

Note that routine to display a torus is modified
relative to the official three.js release to allow more parameters, the modified
version is included in MARXS.

For reference, here is a short summary of the json data layout.
There are two main entries: ``metadata`` (a list with the MARXS version, writer, etc...)
and ``objects`` which is a list of lists.
Each of the sublists has the following fields:

- ``n``: number of objects in list
- ``name``: string or list (if there are sevaral object in this sublist)
- ``material``: string
- ``materialpropterties``: dict
- ``geometry``: type (e.g. ``BufferGeometry``, see `three.js <threejs.org>`_)
- if ``geometry``  is a buffer Geometry: ``pos``, ``color`` (lists of numbers)
- otherwise:
  - ``pos4d``: list of lists of 16 numbers
  - ``geometrypars``: list (meaning depends on geometry, e.g. radius of torus)
'''
import os
import json
import datetime
import warnings
import numpy as np
from astropy.utils.decorators import format_doc


from . import threejs
from . import utils
from marxs import __version__ as version

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

doc_plot='''
    {__doc__}

    Parameters
    ----------
    obj : `marxs.base.MarxsElement`
        The element that should be plotted.
    display : dict of None
        Dictionary with display settings.

    Returns
    -------
    outjson : dict
        ``outjson`` is a (possibly nested)  dictionaries that describes the scene
        in form that the included MARXSloader.js can read.
'''


@format_doc(doc_plot)
def plot_object(obj, display=None, **kwargs):
    ''''Format any MARXS object as a json string.

    This method will inspect the object that is passed in and select the
    correct plotting method for its shape.
    '''
    return utils.plot_object_general(plot_registry, obj, display=display, **kwargs)


def container(obj, display=None,  **kwargs):
    ''''Recursively output three.js json to describe all elements of a container.

    Output of each element can be a dict (if it is a leaf) or a list
    (if it is a container). We need to flatten the list here to avoid
    arbitrarily deep recursion.
    '''
    out = []
    for elem in obj.elements:
        elemout = plot_obj(elem, elem.display)
        if isinstance(elemout, list):
            out.extend(elemout)
        elif (elemout is None):
            pass
        else:
            out.append(elemout)
    return out


def box(obj, display):
    '''Describe a box-shaped optical elements.'''
    out = {}
    out['n'] = 1
    out['name'] = str(obj.name)
    out['material'] = 'MeshStandardMaterial'
    out['materialproperties'] = threejs.materialdict(display, out['material'])
    out['geometry'] = 'BoxGeometry'
    out['geometrypars'] = (2, 2, 2)
    out['pos4d'] = [obj.pos4d.T.flatten().tolist()]
    if not ('side' in display):
        out['materialproperties']['side'] = 2
    return out


def triangulation(obj, display):
    '''Describe a plane with a hole, such as an aperture of baffle.'''
    xyz, triangles = obj.geometry.triangulate(display)
    out = {}
    out['n'] = 1
    out['name'] = str(obj.name)
    out['material'] = 'MeshStandardMaterial'
    out['materialproperties'] = threejs.materialdict(display, out['material'])
    out['geometry'] = 'BufferGeometry'
    out['geometrytype'] = 'Mesh'
    out['pos'] = [xyz.flatten().tolist()]
    out['faces'] = [triangles.flatten().tolist()]

    if not ('side' in display):
        out['materialproperties']['side'] = 2

    return out


def torus(obj, display, theta0=0., thetaarc=2*np.pi, phi0=0., phiarc=np.pi * 2):
    '''Describe a (possibly incomplete) torus.

    The plot range for theta and phi is taken from the values for ``coo1`` and ``coo2``
    in the ``display`` dictionary. There entries should be a list of value if less then
    the full torus (ranging from 0 to 2 pi in each coordinate) is desired for the plot.
    '''
    theta = display.get('coo1', [0, 2 * np.pi])
    phi = display.get('coo2', [0, 2 * np.pi])

    out = {}
    out['n'] = 1
    out['name'] = str(obj.name)
    out['material'] = 'MeshStandardMaterial'
    out['materialproperties'] = threejs.materialdict(display, out['material'])
    out['geometry'] = 'ModifiedTorusBufferGeometry'
    out['geometrypars'] = (obj.R, obj.r,
                           int(np.rad2deg(theta[1])), int(np.rad2deg(phi[1])),
                           theta[1], theta[0], phi[1], phi[0])
    out['pos4d'] = [obj.pos4d.flatten().tolist()]

    if not ('side' in display):
        out['materialproperties']['side'] = 2

    return out


def plot_rays(data, scalar=None, prop={}, name='Photon list', cmap=None):
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
    prop : dict
        keyword arguments for line material.
    name : string
        Identifier "name" for three.js objects. This only matters if your website
        identifies elements by name for interactive features.
    cmap : `matplotlib.colors.Colormap` instance or string or None
        `matplotlib` color maps are used to convert ``scalar`` values to rgb colors.
        If ``None`` the default matplotlib colormap will be used, otherwise the colormap
        can be specified in this keyword.
    '''
    data, s_rgb, prob, n = threejs._format_plot_rays_input(data, scalar, cmap, prop)

    out = {}
    out['n'] = n
    out['name'] = name
    out['material'] = 'LineBasicMaterial'
    out['materialproperties'] = threejs.materialdict(prop, out['material'])
    out['geometry'] = 'BufferGeometry'
    out['geometrytype'] = 'Line'
    out['pos'] = data.reshape((n, -1)).tolist()
    out['color'] = s_rgb[:, :, :3].reshape((n, -1)).tolist()
    return out


def write(fileobject, data, photons=None):
    '''Add metadata and write json for three.js to disk

    Parameters
    ----------
    fileobject : writeable file-like object
    data : list of dict or single dict
        Output of ``xxx.plot(format='threejsjson')`` calls. This can either
        be a list of dictionaries or a single dictionary.
    photons : `astropy.table.Table` or None
        Some metadata is copied from a photon list, if available.
    '''
    if not isinstance(data, list):
        data = [data]
    date = datetime.datetime.now()
    jdata = {'meta': {'format_version': 1,
                      'origin': 'MARXS:threejsjson output',
                      'date': str(date.date()),
                      'time': str(date.time()),
                      'marxs_version': version},
             'elements': data}

    if photons is not None:
        data['runinfo'] = photons.meta()

    if HAS_JSONSCHEMA:
        path =  os.path.abspath(os.path.dirname(__file__))
        schemafile = os.path.join(path, 'threejs_files', 'jsonschema.json')
        with open(schemafile) as f:
            schema = json.load(f)
        jsonschema.validate(jdata, schema)

    else:
        warnings.warn('Module jsonschema not installed. json file will be written without further verification.')

    json.dump(jdata, fileobject)


plot_registry = {'triangulation': triangulation,
                 'torus': torus,
                 'box': box,
                 'container': container,
                 }
