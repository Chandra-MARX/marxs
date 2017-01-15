import os
import json
import datetime
import numpy as np

from ..version import version
from . import threejs
from .utils import format_saved_positions, plot_object_general

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


def plot_object(obj, display=None, **kwargs):
    return plot_object_general(plot_registry, obj, display=display, **kwargs)


def container(obj, **kwargs):
    '''Output of each element can be a dict (if it is a leaf) or a list
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


def plane_with_hole(obj, display):
    xyz, triangles = obj.triangulate_inner_outer()
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
    out = {}
    out['n'] = 1
    out['name'] = str(obj.name)
    out['material'] = 'MeshStandardMaterial'
    out['materialproperties'] = threejs.materialdict(display, out['material'])
    out['geometry'] = 'ModifiedTorusBufferGeometry'
    out['geometrypars'] = (obj.R, obj.r,
                           int(np.rad2deg(thetaarc)), int(np.rad2deg(phiarc)),
                           thetaarc, theta0, phiarc, phi0)
    out['pos4d'] = [obj.pos4d.flatten().tolist()]

    if not ('side' in display):
        out['materialproperties']['side'] = 2

    return out


def plot_rays(data, scalar=None, prop={}, name='Photon list', cmap=None):
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


plot_registry = {'plane with hole': plane_with_hole,
                 'torus': torus,
                 'box': box,
                 'container': container,
                 }
