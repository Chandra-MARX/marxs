# Licensed under GPL version 3 - see LICENSE.rst
def get_package_data():
    # I could not figure out how to get package_data
    # to deal with a directory hierarchy of files, so just explicitly list.
    return {
        'marxs.missions.chandra': ['HESSdesign.rdb',
                                  'tests/sim_asol.fits',
                         ]
    }
