#!/usr/bin/env python
import sys
# To use a consistent encoding
from codecs import open
from os import path

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
from sphinx.setup_command import BuildDoc

import versioneer


### import DESCRIPTION file ###
here = path.abspath(path.dirname(__file__))
# Get the long description from the relevant file
with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
    long_description = f.read()

### versioneer ###
versioneer.VCS = 'git'
versioneer.versionfile_source = 'marxs/_version.py'
versioneer.versionfile_build = 'marxs/_version.py'
versioneer.tag_prefix = '' # tags are like 1.2.0
versioneer.parentdir_prefix = 'marxs-' # dirname like 'myproject-1.2.0'

### py.test ###
class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


cmdclass = versioneer.get_cmdclass()
cmdclass['test'] = PyTest
cmdclass['build_sphinx'] = BuildDoc


setup(name='marxs',
      description='Multi Architecture Raytracer for X-ray satellites',
      long_description=long_description,
      author='H. M. Guenther',
      author_email='hgunther@mit.edu',

      version=versioneer.get_version(),
      cmdclass=cmdclass,

      license='GPLv3+',

      # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        'Development Status :: 2 - Pre-Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy'
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        # Specify the Python versions you support here. In particular, ensure.
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        ],

      packages = find_packages(),

      setup_requires=["cffi>=1.0.0", "astropy>=1.0"],
      cffi_modules=["marxs/optics/marx_build.py:ffi"],
      install_requires=["cffi>=1.0.0"],
      tests_require=['pytest'],
      zip_safe=False,
     )
