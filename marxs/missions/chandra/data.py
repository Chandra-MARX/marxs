# Licensed under GPL version 3 - see LICENSE.rst
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
'''Constants used to transform from chip coordinates to TDET system.

Numbers are taken for the Chandra coordinate memo I at
http://cxc.harvard.edu/contrib/jcm/ncoords.ps
'''

ODET = {'ACIS': [4096.5, 4096.5]}
'''Position of coordinate origin in detector coordinates'''

PIXSIZE = {'ACIS': 0.492 / 3600.}
'''Nominal size of pixels on the sky in degrees.'''

PIX_CORNER_LSI_PAR='''
###################  LSI to CPC corners definition file ##############
#   Chip Corner LSI coords of ACIS, HRC-I, HSI, and ACIS-2C in mm    #
#                                                                    #
#   Chip  CPC XMAX   CPC YMAX                                        #
#   Corner   LSI X  LSI Y    LSI Z                                   #
#                                                                    #
# Reference: Tables 14-17, "ASC Coordinates", July 31, 1996 version  #
######################################################################

ACIS,s,h,"CORNERS AXAF-LSI-1.0/ACIS-I&S&C AXAF-CPC-1.0",,,
#====================================================================#
#
ACIS-Title,s,h,"ACIS-I/S (AI1) LSI coordinates (mm)",,,
ACIS-chip-size,s,h,"(1024, 1024)",,," ACIS chip size (x, y) in pixel"
#
ACIS-I0-LL,s,h,"( 2.361 -26.484  23.088 )",,,"LSI coords for ACIS-I0 LL"
ACIS-I0-LR,s,h,"( 1.130 -26.546  -1.458 )",,,"LSI coords for ACIS-I0 LR"
ACIS-I0-UR,s,h,"(-0.100  -2.001  -1.458 )",,,"LSI coords for ACIS-I0 UR"
ACIS-I0-UL,s,h,"( 1.130  -1.939  23.088 )",,,"LSI coords for ACIS-I0 UL"
#
ACIS-I1-LL,s,h,"( 1.130  23.086  -1.458 )",,,"LSI coords for ACIS-I1 LL"
ACIS-I1-LR,s,h,"( 2.360  23.024  23.088 )",,,"LSI coords for ACIS-I1 LR"
ACIS-I1-UR,s,h,"( 1.130  -1.521  23.088 )",,,"LSI coords for ACIS-I1 UR"
ACIS-I1-UL,s,h,"(-0.100  -1.459  -1.458 )",,,"LSI coords for ACIS-I1 UL"
#
ACIS-I2-LL,s,h,"( 1.130 -26.546  -1.997 )",,,"LSI coords for ACIS-I2 LL"
ACIS-I2-LR,s,h,"( 2.361 -26.484 -26.543 )",,,"LSI coords for ACIS-I2 LR"
ACIS-I2-UR,s,h,"( 1.130  -1.939 -26.543 )",,,"LSI coords for ACIS-I2 UR"
ACIS-I2-UL,s,h,"(-0.10   -2.001  -1.997 )",,,"LSI coords for ACIS-I2 UL"
#
ACIS-I3-LL,s,h,"( 2.361  23.024 -26.543 )",,,"LSI coords for ACIS-I3 LL"
ACIS-I3-LR,s,h,"( 1.131  23.086  -1.997 )",,,"LSI coords for ACIS-I3 LR"
ACIS-I3-UR,s,h,"(-0.100  -1.459  -1.997 )",,,"LSI coords for ACIS-I3 UR"
ACIS-I3-UL,s,h,"( 1.13   -1.521 -26.543 )",,,"LSI coords for ACIS-I3 UL"
#
ACIS-S0-LL,s,h,"( 0.744 -81.170 -59.170 )",,,"LSI coords for ACIS-S0 LL"
ACIS-S0-LR,s,h,"( 0.353 -56.597 -59.170 )",,,"LSI coords for ACIS-S0 LR"
ACIS-S0-UR,s,h,"( 0.353 -56.597 -34.590 )",,,"LSI coords for ACIS-S0 UR"
ACIS-S0-UL,s,h,"( 0.744 -81.170 -34.590 )",,,"LSI coords for ACIS-S0 UL"
#
ACIS-S1-LL,s,h,"(0.348  -56.133 -59.170 )",,,"LSI coords for ACIS-S1 LL"
ACIS-S1-LR,s,h,"(0.099  -31.559 -59.170 )",,,"LSI coords for ACIS-S1 LR"
ACIS-S1-UR,s,h,"(0.099  -31.559 -34.590 )",,,"LSI coords for ACIS-S1 UR"
ACIS-S1-UL,s,h,"(0.348  -56.133 -34.590 )",,,"LSI coords for ACIS-S1 UL"
#
ACIS-S2-LL,s,h,"( 0.096 -31.100 -59.170 )",,,"LSI coords for ACIS-S2 LL"
ACIS-S2-LR,s,h,"(-0.011  -6.524 -59.170 )",,,"LSI coords for ACIS-S2 LR"
ACIS-S2-UR,s,h,"(-0.011  -6.524 -34.590 )",,,"LSI coords for ACIS-S2 UR"
ACIS-S2-UL,s,h,"( 0.096 -31.100 -34.590 )",,,"LSI coords for ACIS-S2 UL"
#
ACIS-S3-LL,s,h,"(-0.011  -6.035 -59.170 )",,,"LSI coords for ACIS-S3 LL"
ACIS-S3-LR,s,h,"( 0.024  18.541 -59.170 )",,,"LSI coords for ACIS-S3 LR"
ACIS-S3-UR,s,h,"( 0.024  18.541 -34.590 )",,,"LSI coords for ACIS-S3 UR"
ACIS-S3-UL,s,h,"(-0.011  -6.035 -34.590 )",,,"LSI coords for ACIS-S3 UL"
#
ACIS-S4-LL,s,h,"( 0.026  18.970 -59.170 )",,,"LSI coords for ACIS-S4 LL"
ACIS-S4-LR,s,h,"( 0.208  43.545 -59.170 )",,,"LSI coords for ACIS-S4 LR"
ACIS-S4-UR,s,h,"( 0.208  43.545 -34.590 )",,,"LSI coords for ACIS-S4 UR"
ACIS-S4-UL,s,h,"( 0.026  18.970 -34.590 )",,,"LSI coords for ACIS-S4 UL"
#
ACIS-S5-LL,s,h,"( 0.208  43.986 -59.170 )",,,"LSI coords for ACIS-S5 LL"
ACIS-S5-LR,s,h,"( 0.528  68.560 -59.170 )",,,"LSI coords for ACIS-S5 LR"
ACIS-S5-UR,s,h,"( 0.528  68.560 -34.590 )",,,"LSI coords for ACIS-S5 UR"
ACIS-S5-UL,s,h,"( 0.208  43.986 -34.590 )",,,"LSI coords for ACIS-S5 UL"
#
#
HRC,s,h,"CORNERS AXAF-LSI-1.0/HRC-I&S AXAF-CPC-1.0",,,
#
#====================================================================#
#
#Updated on 12/24/97
HRC-Title,s,h,"HRC LSI coordinates (mm)",,,
HRC-I-chip-size,s,h,"(16384, 16384)",,,"HRC-I chip size (x, y) in pixel"
HRC-S-chip-size,s,h," ( 4096, 16456)",,," HRC-S1 & S3 chip size (x, y) in pixel"
HRC-S2-chip-size,s,h,"( 4096, 16456)",,,"HRC-S2 chip size (x, y) in pixel"
#
HRC-I-LL,s,h,"(0.0      0.000   74.482 )",,,"LSI coords for HRC-I LL"
HRC-I-LR,s,h,"(0.0     74.482    0.000 )",,,"LSI coords for HRC-I LR"
HRC-I-UR,s,h,"(0.0      0.000  -74.482 )",,,"LSI coords for HRC-I UR"
HRC-I-UL,s,h,"(0.0    -74.482    0.000 )",,,"LSI coords for HRC-I UL"
#
HRC-S1-LL,s,h,"(2.644  161.949  -13.167 )",,,"LSI coords for HRC-S1 LL"
HRC-S1-LR,s,h,"(2.644  161.949   13.167 )",,,"LSI coords for HRC-S1 LR"
HRC-S1-UR,s,h,"(0.000   56.180   13.167 )",,,"LSI coords for HRC-S1 UR"
HRC-S1-UL,s,h,"(0.000   56.180  -13.167 )",,,"LSI coords for HRC-S1 UL"
#
HRC-S2-LL,s,h,"(0.000   56.180  -13.167 )",,,"LSI coords for HRC-S3 LL"
HRC-S2-LR,s,h,"(0.000   56.180   13.167 )",,,"LSI coords for HRC-S3 LR"
HRC-S2-UR,s,h,"(0.000  -49.622   13.167 )",,,"LSI coords for HRC-S3 UR"
HRC-S2-UL,s,h,"(0.000  -49.622  -13.167 )",,,"LSI coords for HRC-S3 UL"
#
HRC-S3-LL,s,h,"(0.000  -49.622  -13.167 )",,,"LSI coords for HRC-S2 LL"
HRC-S3-LR,s,h,"(0.000  -49.622   13.167 )",,,"LSI coords for HRC-S2 LR"
HRC-S3-UR,s,h,"(2.253 -155.400   13.167 )",,,"LSI coords for HRC-S2 UR"
HRC-S3-UL,s,h,"(2.253 -155.400  -13.167 )",,,"LSI coords for HRC-S2 UL"
'''


def chip2tdet(chip, tdet, id_num):
    '''Convert CHIP coordinates to TDET coordiantes.

    See eqn (5) in `the Chandra coordiante memo <http://cxc.harvard.edu/contrib/jcm/ncoords.ps>`_.

    Parameters
    ----------
    chip : (N, 2) np.array
        chip coordiantes
    tdet : dict
        dictionary with definitions for the coordiante conversion.
        See `ACISTDET` for an example.
    id_num : integer
        chip ID number (e.g. ``1`` for ACIS-I1)
    '''
    scale = tdet['scale'][id_num]
    handedness = tdet['handedness'][id_num]
    origin_tdet = tdet['origin'][id_num]
    theta = tdet['theta'][id_num]
    rotation = np.array([[np.cos(theta), np.sin(theta)],
                         [-np.sin(theta), np.cos(theta)]])
    return scale * handedness * np.dot(rotation, (chip - 0.5).T).T + (origin_tdet + 0.5)
