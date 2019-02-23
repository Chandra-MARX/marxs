# Licensed under GPL version 3 - see LICENSE.rst
def get_package_data():
    # I could not figure out how to get package_data
    # to deal with a directory hierarchy of files, so just explicitly list.
    return {
        'marxs.missions.mitsnl': ['SiTransmission.csv',
                                  'tests/grating_efficiency.csv',
                                  'tests/grating_efficiency_broken.csv',
                         ]
    }
