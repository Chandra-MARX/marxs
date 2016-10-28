import numpy as np

from marxs.math.pluecker import h2e
from .utils import color_tuple_to_hex

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
    spec = []
    for k in display:
        if (k in listofproperties['Material']) or (k in listofproperties[material]):
            # now special cases that need to be transformed in some way
            if k == 'color':
                spec.append('{0} : {1}'.format(k, color_tuple_to_hex(display[k])))
            else:
                spec.append('{0} : {1}'.format(k, display[k]))
    if ('opacity' in display) and not ('transparent' in display):
        spec.append('transparent : true')
    if not ('side' in display):
        spec.append('side : THREE.DoubleSide')
    return ', '.join(spec)


def plot_rays(data, outfile, scalar=None, cmap=None,
              prop={}):
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
    outfile : file object
        Output javascript code is written to this file.
    prop : dict
        keyword arguments for line material.
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
        raise ValueError('Scalar quantity for each point must have shape ({0},) or ({0}, {1})'.format(n, N))

    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap)
    s_rgb = cmap(s)

    materialspec = materialspec(prop, 'LineBasicMaterial')

    for i in range(n):
        positions = ', '.join(np.asarray(data[i, :, :], dtype=str).flatten())
        colors = ', '.join(np.asarray(s_rgb[i, :, :], dtype=str).flatten())
        outfile.write('''
        var geometry = new THREE.BufferGeometry();
	var material = new THREE.LineBasicMaterial({{ {materialspec} }});
	var positions = new Float32Array([{positions}]);
	var colors = new Float32Array([{colors}]);
	geometry.addAttribute( 'position', new THREE.BufferAttribute( positions, 3 ) );
	geometry.addAttribute( 'color', new THREE.BufferAttribute( colors, 3 ) );
	geometry.computeBoundingSphere();
	mesh = new THREE.Line( geometry, material );
	scene.add( mesh );'''.format(positions=positions, colors=colors, materialspec=materialspec))
