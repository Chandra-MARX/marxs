#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from setuptools import setup

# To get MARX binary info
from configparser import ConfigParser, NoOptionError


# check is MARX C code is configured in setup.py and if it is, add to
# appropriate arguments to setup args
conf = ConfigParser()
conf.read(['setup.cfg'])
setup_args = {}
try:
    marxscr = conf.get('MARX', 'srcdir')
    marxlib = conf.get('MARX', 'libdir')
    if marxscr and marxlib:
        setup_args['cffi_modules'] = ["marxs/optics/marx_build.py:ffi"]
        print('Using MARX C code at {0}\n and compiled libraries at {1}'.format(marxscr, marxlib))
    else:
        print('MARX C code is not configured in setup.cfg - modules will be disabled.')
except NoOptionError:
    print('MARX C code is not configured in setup.cfg - modules will be disabled.')


setup(**setup_args)
