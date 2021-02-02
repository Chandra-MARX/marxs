# Licensed under GPL version 3 - see LICENSE.rst
'''This module collects helper functions for visualization backends.

The functions here are not intended to be called directly by the
user. Instead, they refractor common tasks that are used in several
visualization backends.
'''
import warnings
import numpy as np


class MARXSVisualizationWarning(Warning):
    '''Warning class for MARXS objects missing from plotting'''
    pass


def get_obj_name(obj):
    '''Return printable name for objects or functions.'''
    if hasattr(obj, 'name'):
        return obj.name
    elif hasattr(obj, 'func_name'):
        return obj.func_name
    else:
        return str(obj)


class DisplayDict(dict):
    '''A dictionary to store how an element is displayed in plotting.

    A dictionary of this type works just like a normal dictionary,
    except for an additional look-up step for keys that are not found
    in the dictionary itself.  A ``DisplayDict`` is initialized with a
    reference to the object it describes and any parameter accessed
    from ``DisplayDict`` that is not found in the dictionary will be
    searched for in the object's geometry. This allows us to set any
    and all display settings in the ``DisplayDict`` to customize
    plotting in any way without affecting how the ray-trace is run
    (which uses only the parameters set in the geoemtry), but for
    those values that are not set, fall back to the settings of the
    geometry (e.g. the shape of an object is typically taken from the
    geometry, while the color is not).

    Parameters
    ----------
    parent : `marxs.base.MarxsElement`
        Reference to the object that is described by this ``DisplayDict``
    args, kwargs: see `dict`

    '''
    def __init__(self, parent, *args, **kwargs):
        self.parent = parent
        super(DisplayDict, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        if (key not in self) and hasattr(self.parent, 'geometry'):
            try:
                return getattr(self.parent.geometry, key)
            except AttributeError:
                raise KeyError(key)
        else:
            return super(DisplayDict, self).__getitem__(key)

    def get(self, k, d=None):
        '''D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.'''
        try:
            return self[k]
        except:
            KeyError
            return d


def plot_object_general(plot_registry, obj, display=None, **kwargs):
    '''Look up a plotting routine for an object and execute it.

    This function is not meant to be called directly by the user, instead, it
    is designed to simplify the implementation of new plotting backends.

    Parameters
    ----------
    plot_registry : dict
        Keys are the names of the shape of an object and values in this
        dictionary are functions that know how to plot this type of shape. The
        appropriate plotting function is then called with the input `obj`,
        `display` and any other keyword arguments.
        If the shape is ``"None"`` (as a string), no plotting function is
        called.
    obj : `marxs.base.MarxsElement`
        The element that should be plotted.
    display : dict of None
        Dictionary with display settings. If this is ``None``, ``obj.display``
        is used. If that is also ``None`` then the objects is skipped.
    kwargs : other keyword arguments
        These arguments are just passed through to the plotting function.

    Returns
    -------
    out : backend-dependent
        The output from the plotting function that was executed is passed
        through.  Different plotting backends return different kinds of output.
    '''
    if display is None:
        if hasattr(obj, 'display') and (obj.display is not None):
            display = obj.display
        else:
            warnings.warn('Skipping {0}: No display dictionary found.'.format(
                get_obj_name(obj)),
                          MARXSVisualizationWarning)
            return None

    try:
        shape = display['shape']
    except KeyError:
        warnings.warn('Skipping {0}: "shape" not set in display dict.'.format(
            get_obj_name(obj)),
                      MARXSVisualizationWarning)
        return None

    shapes = [s.strip() for s in shape.split(';')]
    for s in shapes:
        if s == 'None':
            return None
        elif s in plot_registry:
            # turn into valid color tuple
            display['color'] = get_color(display)
            return plot_registry[s](obj, display,  **kwargs)
    else:
        warnings.warn('Skipping {0}: No function to plot {1}.'.format(
            get_obj_name(obj), shape),
                      MARXSVisualizationWarning)
        return None


def get_color(d):
    '''Look for color information in dictionary.

    If missing, return white.
    This function checks if the `d['color']` is a valid RGB tuple and if
    not it imports a `matplotlib.colors.ColorConverter` to convert any
    matplotlib compatible string to an RGB tuple.

    Parameters
    ----------
    d : dict
        Color information should be present in ``d['color']``.

    Returns
    -------
    color : tuple
        RGB tuple with each element in the range 0..1
    '''
    if 'color' not in d:
        return (1., 1., 1.)
    else:
        c = d['color']
        # check if this is a tuple of three floats n the range 0..1
        # or can be converted to one
        try:
            cout = tuple(c)
            if len(cout) != 3:
                raise TypeError
            for a in cout:
                if not isinstance(a, float) or (a < 0.) or (a > 1.):
                    raise TypeError
            return cout
        except TypeError:
            # It's a hex or string. Let matplotlib deal with that.
            import matplotlib.colors
            return matplotlib.colors.colorConverter.to_rgb(c)


def color_tuple_to_hex(color):
    '''Convert color tuple to hex string.

    Parameters
    ----------
    color : tuple
        tuple has three elements (rgb) that are floats betwen 0 and 1
        or ints between 0 and 255.

    Returns
    -------
    hexstring : string
        string encoding that number as hex
    '''
    if all([isinstance(a, float) for a in color]):
        if any(i < 0. for i in color) or any(i > 1. for i in color):
            raise ValueError('Float values in color tuple must be between 0 and 1.')
        out = hex(int(color[0] * 256**2 * 255 +
                      color[1] * 256 * 255 +
                      color[2] * 255))
    elif all([isinstance(a, int) for a in color]):
        if any(i < 0 for i in color) or any(i > 255 for i in color):
            raise ValueError('Int values in color tuple must be between 0 and 255.')
        out = hex(color[0] * 256**2 + color[1] * 256 + color[2])
    else:
        raise ValueError('Input tuple must be all float or all int.')
    # Now pad with zeros if required
    return out[:2] + out[2:].zfill(6)


def plane_with_hole(outer, inner):
    '''Triangulation of a plane with an inner hole

    This function constructs a triangulation for a plane with an inner
    hole, e.g. a rectangular plane where an inner circle is cut out.

    Parameters
    ----------
    outer, inner : np.ndarray of shape (n, 3)
        Coordinates in x,y,z of points that define the inner and outer
        boundary.  ``outer`` and ``inner`` can have a different number of
        points, but points need to be listed in the same orientation
        (e.g. clockwise) for both and the starting points need to have a
        similar angle as seen from the center (e.g. for a plane with z=0, both
        ``outer`` and ``inner`` could list a point close to the y-axis first.

    Returns
    -------
    xyz : nd.array
        stacked ``outer`` and ``inner``.
    triangles : nd.array
        List of the indices. Each row has the index of three points in ``xyz``.

    Examples
    --------
    In this example, we make a square and cut out a smaller square in the
    middle.

    >>> import numpy as np
    >>> from marxs.visualization.utils import plane_with_hole
    >>> outer = np.array([[-1, -1, 1, 1], [-1, 1, 1, -1], [0,0,0,0]]).T
    >>> inner = 0.5 * outer
    >>> xyz, triangles = plane_with_hole(outer, inner)
    >>> triangles
    array([[0, 4, 5],
       [0, 1, 5],
       [1, 5, 6],
       [1, 2, 6],
       [2, 6, 7],
       [2, 3, 7],
       [3, 7, 4],
       [3, 0, 4]])

    '''
    n_out = outer.shape[0]
    n_in = inner.shape[0]
    n = n_out + n_in

    triangles = np.zeros((n, 3), dtype=int)
    xyz = np.vstack([outer, inner])

    i_in = 0
    i_out = 0
    for i in range(n_out + n_in):
        if i/n >= i_in/n_in:
            triangles[i, :] = [i_out, n_out + i_in,
                               n_out + ((i_in + 1) % n_in)]
            i_in += 1
        else:
            triangles[i, :] = [i_out, (i_out + 1) % n_out,
                               n_out + (i_in % n_in)]
            i_out = (i_out + 1) % n_out
    return xyz, triangles


def combine_disjoint_triangulations(list_xyz, list_triangles):
    '''Combine two disjoint triangulations into one set of points

    This function combines two entirely separate triangulations into
    one set of point and triangles. Plotting the combined
    triangulation should have the same effect as plotting each
    triangulation separately. This function is used for plotting
    apertures where we have e.g. an open ring. This can be plotted as
    an inner circle plus an outer shape with a hole in it.

    Parameters
    ----------
    list_xyz : list of `np.array`
        Each array holds xyz values for one triangulation
    list_triangles : list of nd.array
        Each array holds the list of the indices for one triangulation.

    Returns
    -------
    xyz : nd.array
        stacked ``outer`` and ``inner``.
    triangles : nd.array
        List of the indices. Each row has the index of three points in ``xyz``.

    '''
    xyz = np.vstack(list_xyz)
    n_offset = np.cumsum([a.shape[0] for a in list_xyz])
    n_offset -= n_offset[0]
    triangles = np.vstack([list_triangles[i] + n_offset[i] for i in
                           range(len(n_offset))])
    return xyz, triangles
