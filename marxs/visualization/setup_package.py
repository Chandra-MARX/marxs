def get_package_data():
    # Installs the data files.  Unable to get package_data
    # to deal with a directory hierarchy of files, so just explicitly list.
    return {
        'marxs.visualization': ['threejs_files/MARXSloader.js',
                                'threejs_files/loader.html',
                                'threejs_files/ModifiedTorusBufferGeometry.js',
                                'threejs_files/jsonschema.json'
                                   ]
    }
