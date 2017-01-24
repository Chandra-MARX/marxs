'''Plot routines for display with `three.js <http://threejs.org>`__.

Each routine adds strings to a file and thus each routing requires a writable file
object as an argument. This file can then be included in a webpage that loads
the three.js javascript package.

Note that routine to display a torus is modified
relative to the official three.js release to allow more parameters; the modified
version is included in the MARXS source code.
Also, note that the output file is not a valid webside by itself. Instead, it is a list of
javascript commands, that needs to be included into a website after the nesessary setup
and package loading.
'''

import numpy as np
from astropy.utils.decorators import format_doc

from . import utils

listofproperties = {'Material': ['id', 'name', 'opacity', 'transparent', 'blending', 'blendSrc',
                                 'blendDst', 'BlendEquation', 'depthTest', 'depthWrite',
                                 'polygonOffset', 'plygonOffsetFactor', 'polygonOffsetUnits',
                                 'alphaTest', 'clippingPlanes', 'clipShadows', 'overdraw',
                                 'visible', 'side', 'needsUpdate'],
                    'MeshStandardMaterial': ['color', 'roughness', 'metalness', 'map', 'lightMap',
                                             'lightMapIntensity', 'aoMap', 'aoMapIntensity',
                                             'emissive', 'emissiveMap', 'emissiveIntensity',
                                             'bumpMap', 'bumpMapScale', 'normalMap', 'normalMapScale',
                                             'displacementMap', 'displacementScale',
                                             'displacementBias', 'roughnessMap', 'metalnessMap',
                                             'alphaMap', 'envMap', 'envMapIntensity', 'refractionRatio',
                                             'fog', 'shading', 'wireframe', 'wireframeLinewidth',
                                             'wireframeLinecap', 'wireframeLinejoin', 'vertexColors',
                                             'skinning', 'morphTargets', 'morphNormals'],
                    'LineBasicMaterial' : ['color', 'linewidth', 'linecap', 'linejoin', 'vertexColors',
                                           'fog'],
                    }


def array2string(array):
    '''Flatten array and return string representation for insertion into js.'''
    return np.array2string(array.flatten(), max_line_width=1000, separator=',',
                           formatter={'float_kind': '{0:.1f}'.format})

def materialdict(display, material):
    '''Construct a string that can be pasted into a javascript template to describe a three.js material.

    Parameters
    ----------
    display : dict
        Dictionary of properties. All properties that have a same name as the property of
        the relevant material in javascript are included, all others are ignored.
    material : string
        Name of the material in three.js (Only materials used currently in marxs are supported).

    Returns
    -------
    spec : dict
        material specification
    '''
    spec = {}
    for k in display:
        if (k in listofproperties['Material']) or (k in listofproperties[material]):
            # now special cases that need to be transformed in some way
            if k == 'color':
                spec['color'] = utils.color_tuple_to_hex(display[k]).replace('0x', '#')
            else:
                spec[k] = display[k]
    if ('opacity' in display) and not ('transparent' in display):
        spec['transparent'] = 'true'
    return spec


def materialspec(display, material):
    '''Construct a string that can be pasted into a javascript template to describe a three.js material.

    Parameters
    ----------
    display : dict
        Dictionary of properties. All properties that have a same name as the property of
        the relevant material in javascript are included, all others are ignored.
    material : string
        Name of the material in three.js (Only materials used currently in marxs are supported).

    Returns
    -------
    spec : string
        String that can be pasted into javasript files.
    '''
    matdict = materialdict(display, material)
    spec = ['{0} : {1}'.format(k, matdict[v]) for k in matdict]
    return ', '.join(spec)

def _format_plot_rays_input(data, scalar, cmap, prop):
    if hasattr(data, 'data') and isinstance(data.data, list):
        data = utils.format_saved_positions(data)

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
        prop['vertexColors'] = 2 # 2 = 'THREE.VertexColors'

    return data, s_rgb, prop, n

def plot_rays(data, outfile, scalar=None, cmap=None,
              prop={}):
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
    cmap : `matplotlib.colors.Colormap` instance or string or None
        `matplotlib` color maps are used to convert ``scalar`` values to rgb colors.
        If ``None`` the default matplotlib colormap will be used, otherwise the colormap
        can be specified in this keyword.
    '''
    data, s_rgb, prop, n = _format_plot_rays_input(data, scalar, cmap, prob)
    material = materialspec(prop, 'LineBasicMaterial')

    for i in range(n):
        positions = array2string(data[i, :, :])
        colors = array2string(s_rgb[i, :, :3])
        outfile.write('''
        var geometry = new THREE.BufferGeometry();
        var material = new THREE.LineBasicMaterial({{ {material} }});
        var positions = new Float32Array({positions});
        var colors = new Float32Array({colors});
        geometry.addAttribute( 'position', new THREE.BufferAttribute( positions, 3 ) );
        geometry.addAttribute( 'color', new THREE.BufferAttribute( colors, 3 ) );
        geometry.computeBoundingSphere();
        mesh = new THREE.Line( geometry, material );
        scene.add( mesh );'''.format(positions=positions, colors=colors, material=material))


doc_plot='''
    {__doc__}

    Parameters
    ----------
    obj : `marxs.base.MarxsElement`
        The element that should be plotted.
    display : dict of None
        Dictionary with display settings.
    outfile : file handle
        Output is written to this file.
'''

@format_doc(doc_plot)
def plot_object(obj, display, outfile, **kwargs):
    '''Plot any MARXS object with three.js as backend.

    This method will inspect the object that is passed in and select the
    correct plotting method for its shape. Javascript code to generate the correct
    represenation in three.js is added to the open file ``outfile``.
    '''
    if 'outfile' not in kwargs:
        raise TypeError('Required argument "outfile" missing.')
    out = plot_object_general(plot_registry, obj, display=display,outfile=outfile, **kwargs)
    return out


@format_doc(doc_plot)
def container(obj, display, outfile):
    '''Recursivey output three.js commands to print all elements of a container.

    Output of each element can be a dict (if it is a leaf) or a list
    (if it is a container). We need to flatten the list here to avoid
    arbitrarily deep recursion.
    '''
    return [plot_obj(e, e.display, outfile) for e in obj.elements]

@format_doc(doc_plot)
def box(obs, display, outfile):
    '''Output three.js commands to include a box-shaped optical element.'''
    matrixstring = ', '.join([str(i) for i in obj.pos4d.flatten()])
    if not ('side' in display):
        display['side'] = 'THREE.DoubleSide'
    materialspec = materialspec(display, 'MeshStandardMaterial')
    outfile.write('''
    var geometry = new THREE.BoxGeometry( 2, 2, 2 );
    var material = new THREE.MeshStandardMaterial( {{ {materialspec} }} );
    var mesh = new THREE.Mesh( geometry, material );
    mesh.matrixAutoUpdate = false;
    mesh.matrix.set({matrix});
    scene.add( mesh );'''.format(materialspec=materialspec,
                                 matrix=matrixstring))

@format_doc(doc_plot)
def plane_with_hole(obj, display, outfile):
    '''Output commands for a plane with an inner hole.'''
    xyz, triangles = obj.triangulate_inner_outer()
    materialspec = materialspec(display, 'MeshStandardMaterial')
    outfile.write('// {}\n'.format(obj.name))
    outfile.write('var geometry = new THREE.BufferGeometry(); \n')
    outfile.write('var vertices = new Float32Array([')
    for row in xyz:
        outfile.write('{0}, {1}, {2},'.format(row[0], row[1], row[2]))
    outfile.write(''']);
    // itemSize = 3 because there are 3 values (components) per vertex
    geometry.addAttribute( 'position', new THREE.BufferAttribute( vertices, 3 ) );
    ''')
    outfile.write('var faces = new Uint16Array([')
    for row in triangles:
        outfile.write('{0}, {1}, {2}, '.format(row[0], row[1], row[2]))
    outfile.write(''']);
    // itemSize = 3 because there are 3 values (components) per triangle
    geometry.setIndex(new THREE.BufferAttribute( faces, 1 ) );
    ''')

    outfile.write('''var material = new THREE.MeshStandardMaterial({{ {materialspec} }});
        var mesh = new THREE.Mesh( geometry, material );
        scene.add( mesh );
        '''.format(materialspec=materialspec))

def torus(obj, display, outfile):
    '''Output commands to display (part of) a torus.

    The plot range for theta and phi is taken from the values for ``coo1`` and ``coo2``
    in the ``display`` dictionary. There entries should be a list of value if less then
    the full torus (ranging from 0 to 2 pi in each coordinate) is desired for the plot.
    '''
    theta = display.get('coo1', [0, 2 * np.pi])
    phi = display.get('coo2', [0, 2 * np.pi])
    materialspec = materialspec(self.display, 'MeshStandardMaterial')
    torusparameters = '{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}'.format(obj.R, obj.r,
                                                                      int(np.rad2deg(theta[1])),
                                                                      int(np.rad2deg(phi[1])),
                                                                      theta[1],
                                                                      theta[0],
                                                                      phi[1],
                                                                      phi[0])
    rot = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1.]])
    matrixstring = ', '.join([str(i) for i in np.dot(obj.pos4d, rot).flatten()])

    outfile.write('''
    var geometry = new THREE.ModifiedTorusBufferGeometry({torusparameters});
    var material = new THREE.MeshStandardMaterial({{ {materialspec} }});
    var mesh = new THREE.Mesh( geometry, material );
    mesh.matrixAutoUpdate = false;
    mesh.matrix.set({matrix});
    scene.add( mesh );'''.format(materialspec=materialspec,
                                 torusparameters=torusparameters,
                                 matrix=matrixstring))

plot_registry = {'plane with hole': plane_with_hole,
                 'torus': torus,
                 'box': box,
                 'container': container,
                 }
