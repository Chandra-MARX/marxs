#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import builtins

# Ensure that astropy-helpers is available
import ah_bootstrap  # noqa

from setuptools import setup
from setuptools.config import read_configuration

from astropy_helpers.setup_helpers import register_commands, get_package_info
from astropy_helpers.version_helpers import generate_version_py

# Store the package name in a built-in variable so it's easy
# to get from other parts of the setup infrastructure
builtins._ASTROPY_PACKAGE_NAME_ = read_configuration('setup.cfg')['metadata']['name']

# Create a dictionary with setup command overrides. Note that this gets
# information about the package (name and version) from the setup.cfg file.
cmdclass = register_commands()

# Freeze build information in version.py. Note that this gets information
# about the package (name and version) from the setup.cfg file.
version = generate_version_py()

# Get configuration information from all of the various subpackages.
# See the docstring for setup_helpers.update_package_files for more
# details.
package_info = get_package_info()

# check is MARX C code is configured in setup.py and if it is, add to
# appropriate arguments to setup args
from configparser import ConfigParser
conf = ConfigParser()
conf.read('setup.cfg')
marxscr = conf.get('MARX', 'srcdir')
marxlib = conf.get('MARX', 'libdir')
if marxscr and marxlib:
    package_info['cffi_modules'] = ["marxs/optics/marx_build.py:ffi"]
    package_info['install_requires'] = ["cffi>=1.0.0"]
    package_info['setup_requires'] = ["cffi>=1.0.0"]
    print('Using MARX C code at {0}\n and compiled libraries at {1}'.format(marxscr, marxlib))
else:
    print('MARX C code is not configured in setup.cfg - modules will be disabled.')

setup(version=version, cmdclass=cmdclass, **package_info)
