import os
import configparser
import subprocess
import logging

from marxs.base import TagVersion

__all__ = ['config', 'tabversion', 'id_num_offset',
           'git_hash', 'git_info',
           ]

logger = logging.getLogger(__name__)

id_num_offset = {'1': 0,
                 '2': 1000,
                 '1m': 10000,
                 '2m': 11000}

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
git_hash = b'no arcus cal data found'
git_info = b'no arcus cal data found'
if len(confs_found) > 0:
    logger.info(f'Found arcus.cfg in {confs_found}')
    try:
        gh = subprocess.check_output(["git", "describe", "--always"],
                                      cwd=config['data']['caldb_inputdata'])[:-1]

        gi = subprocess.check_output(['git', 'show', '-s', '--format=%ci',
                                      gh],
                                      cwd=config['data']['caldb_inputdata'])

        git_hash = gh.decode()[:10]
        git_info = gi.decode()[:19]
    except Exception as err:
        logger.warning(f"Error trying to read {config['data']['caldb_inputdata']}: {err}")


tagversion = TagVersion(SATELLIT=('ARCUS', 'placeholder - no name registered with OGIP'),
                        TELESCOP=('ARCUS', 'placeholder - no name registered with OGIP'),
                        INSTRUME=('ARCUSCAM',  'placeholder - no name registered with OGIP'),
                        FILTER=('NONE', 'filter information'),
                        GRATING=('ARCUSCAT',  'placeholder - no name registered with OGIP'),
                        ARCDATHA=(git_hash, 'Git hash of simulation input data'),
                        ARCDATDA=(git_info, 'Commit time of simulation input data'),
)
