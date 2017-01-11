import numpy as np

from marxs.math.pluecker import h2e
from .utils import color_tuple_to_hex, format_saved_positions

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
                spec['color'] = color_tuple_to_hex(display[k]).replace('0x', '#')
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

    def box(obs, display, outfile):
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

    def plane_with_hole(obj, display, outfile):
        xyz, triangles = obj.triangulate_inner_outer()
        materialspec = materialspec(display, 'MeshStandardMaterial')
        outfile.write('// APERTURE\n')
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

    def torus(obj, display, outfile, theta0=0., thetaarc=2*np.pi,
                      phi0=0., phiarc=np.pi * 2):
        materialspec = materialspec(self.display, 'MeshStandardMaterial')
        torusparameters = '{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}'.format(obj.R, obj.r,
                                                                          int(np.rad2deg(thetaarc)),
                                                                          int(np.rad2deg(phiarc)),
                                                                          thetaarc,
                                                                          theta0,
                                                                          phiarc,
                                                                          phi0)
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
