import numpy as np

NOMINAL_FOCALLENGTH = 10061.65
'''Numbers taken from the Chandra coordinate memo I

at http://cxc.harvard.edu/contrib/jcm/ncoords.ps
'''


TLMINMAX = {'ACIS': {'CCD_ID': [0, 9],
                     'CHIPX': [2, 1023],
                     'CHIPY': [2, 1023],
                     'TDETX': [2, 8191],
                     'TDETY': [2, 8191],
                     'DETX': [0.5, 8192.5],
                     'DETY': [0.5, 8192.5],
                     'X': [0.5, 8192.5],
                     'Y': [0.5, 8192.5],
                     'PHA': [0, 36855],
                     'ENERGY': [0, 1e6],  # This is the ACIS detected energy, not the real one
                     'FLTGRADE': [0, 255],
                    }
            }


AIMPOINTS = {'ACIS-I': [-.782, 0, -233.592],
             'ACIS-S': [-.684, 0, -190.133],
             'HRC-I': [-1.040, 0, 126.985],
             'HRC-S': [-1.43, 0, 250.456],
             }
'''Default aimpoints from coordiante memo.

A lot of work has gone into aimpoints since 2001, specifically because they also move on the
detector. For now, we just implement this simple look-up here, but a more detailed implementation
that reaches out to CALDB might be needed in the future.

Numbers are taken for the Chandra coordinate memo I at
http://cxc.harvard.edu/contrib/jcm/ncoords.ps
'''

TDET = {'ACIS': {'version': 'ACIS-2.2',
                 'theta': np.deg2rad(np.array([90., 270., 90., 270., 0, 0, 0, 0, 0, 0])),
                 'scale': np.ones(10),
                 'handedness': np.ones(10), # could be list of arrays
                 'origin': np.array([[3061, 5131],
                                     [5131, 4107],
                                     [3061, 4085],
                                     [5131, 3061],
                                     [ 791, 1702],
                                     [1833, 1702],
                                     [2875, 1702],
                                     [3917, 1702],
                                     [4959, 1702],
                                     [6001, 1702]])
                 },
        }
'''Constant to used to transform from chip coordinates to TDET system.

Numbers are taken for the Chandra coordinate memo I at
http://cxc.harvard.edu/contrib/jcm/ncoords.ps
'''

ODET = {'ACIS': [4096.5, 4096.5]}
'''Position of coordinate origin in detector coordinates'''

PIXSIZE = {'ACIS': 0.492 / 3600.}
'''Nominal size of pixels on the sky in degrees.'''
