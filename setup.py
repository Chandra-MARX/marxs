#!/usr/bin/env python

from setuptools import setup

import versioneer

versioneer.VCS = 'git'
versioneer.versionfile_source = 'marxs/_version.py'
versioneer.versionfile_build = 'marxs/_version.py'
versioneer.tag_prefix = '' # tags are like 1.2.0
versioneer.parentdir_prefix = 'marxs-' # dirname like 'myproject-1.2.0'


setup(name='marxs',
      description='Multi Architecture Raytracer for X-ray satellites',
      author='H. M. Guenther',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      author_email='hgunther@mit.edu',
      packages = ['marxs']
     )
