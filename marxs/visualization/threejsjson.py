import json
import numpy as np

from marxs.math.pluecker import h2e
from .utils import color_tuple_to_hex
from .threejs import listofproperties

def plot_rays(data scalar=None, cmap=None, prop={}):
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
        prop['vertexColors'] = 'THREE.VertexColors'
    material = materialspec(prop, 'LineBasicMaterial')

    out = {}
    out['n'] = n
    out['name'] = self.name
    out['material'] = 'LineBasicMaterial'
    out['materialproperties'] = threejs.materialspec(self.display, out['material'])
    out['geometry'] = 'BufferGeometry'
    out['pos'] = data.reshape((n, -1)).tolist()
    out['color'] = s_rgb.reshape((n, -1)).tolist()
    return out

def write(filename, data, photons=None):

    jdata = {'meta':, {'version': 1,
                       'origin': 'MARXS:threejsjson output',
             'elements': data}
    if photons is not None:
             jdata['run'] = photons.meta()
    with open(filename, 'w' as f}:
        json.dump(jdata, f)
