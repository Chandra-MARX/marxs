"""Configure Test Suite.

This file is used to configure the behavior of pytest when using the Astropy
test infrastructure. It needs to live inside the package in order for it to
get picked up when running the tests inside an interpreter using
marxs.test

"""
import os

from marxs.missions.arcus.utils import config as arcusconfig

try:
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS
    ASTROPY_HEADER = True
except ImportError:
    ASTROPY_HEADER = False


def pytest_configure(config):
    """Configure Pytest with Astropy.

    Parameters
    ----------
        config : pytest configuration
    """
    if ASTROPY_HEADER:

        config.option.astropy_header = True

        # Customize the following lines to add/remove entries from the list of
        # packages for which version numbers are displayed when running the tests.
        PYTEST_HEADER_MODULES.pop('Pandas', None)
        # PYTEST_HEADER_MODULES['scikit-image'] = 'skimage'

        from . import __version__
        packagename = os.path.basename(os.path.dirname(__file__))
        TESTED_VERSIONS[packagename] = __version__


collect_ignore = ["setup.py"]

# Check if we haave the Arcus CALDB installed
# If not, can't collect those tests, because on import, they will already
# try to read from the CALDB
if 'data' not in arcusconfig:
    collect_ignore.append("missions/arcus/tests/test_analyzegrid.py")
    collect_ignore.append("missions/arcus/tests/test_orders.py")
