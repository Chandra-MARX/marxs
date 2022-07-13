import os
import configparser
import subprocess

from marxs.base import TagVersion

__all__ = ['config', 'tabversion']


config = configparser.ConfigParser()
confs_found = config.read(['arcus.cfg',
                           os.path.expanduser('~/.astropy/config/arcus.cfg')
                           ])

# subprocess uses fork internally, which makes a child process
# that essentially makes a copy of everything that python has
# in memory, essentially doubling the memory footprint of python
# at that point. For long running scripts with big simulations that
# can be enough to run out of memory.
# So, just get this info once during the first import and save in
# module-level variables.
git_hash = subprocess.check_output(["git", "describe", "--always"],
                                   cwd=config['data']['caldb_inputdata'])[:-1]

git_info = subprocess.check_output(['git', 'show', '-s', '--format=%ci',
                                    git_hash],
                                   cwd=config['data']['caldb_inputdata'])

tagversion = TagVersion(SATELLIT=('ARCUS', 'placeholder - no name registered with OGIP'),
                        TELESCOP=('ARCUS', 'placeholder - no name registered with OGIP'),
                        INSTRUME=('ARCUSCAM',  'placeholder - no name registered with OGIP'),
                        FILTER=('NONE', 'filter information'),
                        GRATING=('ARCUSCAT',  'placeholder - no name registered with OGIP'),
                        ARCDATHA=(git_hash.decode()[:10], 'Git hash of simulation input data'),
                        ARCDATDA=(git_info.decode()[:19], 'Commit time of simulation input data'),
)
