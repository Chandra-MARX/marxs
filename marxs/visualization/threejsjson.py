import json
import datetime
import numpy as np

from ..version import version
from . import threejs
from .utils import format_saved_positions


def plot_rays(data, scalar=None, cmap=None, prop={}, name='Photon list'):
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
    outfile : file object
        Output javascript code is written to this file.
    prop : dict
        keyword arguments for line material.
    name : string
        Identifier "name" for three.js objects. This only matters if your website
        identifies elements by name for interactive features.
    '''
    if hasattr(data, 'data') and isinstance(data.data, list):
        data = format_saved_positions(data)

    # The number of points per line
    N = data.shape[1]
    # number of lines
    n = data.shape[0]

    if scalar is None:
        s = np.zeros((n,  N))
    elif scalar.shape == (n, ):
        s = np.tile(scalar, (N, 1)).T
    elif scalar.shape == (n, N):
        s = scalar
    else:
        raise ValueError('Scalar quantity for each point must have shape ({0},) or ({0}, {1})'.format(n, N))

    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap)
    normalizer = plt.Normalize()
    s_rgb = cmap(normalizer(s))

    if 'vertexColors' not in prop:
        prop['vertexColors'] = 2 # 'THREE.VertexColors'

    out = {}
    out['n'] = n
    out['name'] = name
    out['material'] = 'LineBasicMaterial'
    out['materialproperties'] = threejs.materialdict(prop, out['material'])
    out['geometry'] = 'BufferGeometry'
    out['geometrytype'] = 'Line'
    out['pos'] = data.reshape((n, -1)).tolist()
    out['color'] = s_rgb.reshape((n, -1)).tolist()
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

    json.dump(jdata, fileobject)
