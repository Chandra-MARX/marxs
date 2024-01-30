'''MARXS supports several backends for 3D plotting; each of which is implemented as a submodule.

To plot using a specific backend, import the relevant submodule.
'''
from astropy import config as _config

class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for my subpackage.
    """
    xyz_precision = _config.ConfigItem(
        1, 'Precision of coordinate output when floating point numbers are written to text, e.g. json. This is used as `np.round(xyz, p)` to reduce the size of text files.')
    color_precision = _config.ConfigItem(
        2, 'Precision for writing colors as tuples to text files, e.g. json. This is used as `np.round(color, p)` to reduce the size of text files.')

# Create an instance for the user
conf = Conf()