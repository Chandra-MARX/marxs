# Licensed under GPL version 3 - see LICENSE.rst
from os import path
from configparser import ConfigParser
from cffi import FFI

with open (path.join(path.dirname(__file__), "cdef.txt"), "r") as myfile:
    cdeftxt=myfile.read()

ffi = FFI()
ffi.cdef(cdeftxt)


conf = ConfigParser()
# When setup.py is run, then setup.cfg is in the current directory
# However, when runngin this in pytest, it's in ../
# So offer both options here (files not found are silently ignored).
conf.read(['setup.cfg', '../setup.cfg', '../../setup.cfg'])
marxscr = conf.get('MARX', 'srcdir')
marxlib = conf.get('MARX', 'libdir')

sources = [('pfile', 'src', 'pfile.c'), ('marx', 'libsrc', 'mirror.c')]
headers = [('pfile', 'src'), ('jdmath', 'src') , ('jdfits', 'src'),
           ('marx', 'src',), ('marx', 'libsrc',), ('src',), ('libsrc',)]

ffi.set_source("_marx",
'''
# include "pfile.h"
# include "_pfile.h"

# include <jdmath.h>
# include "marx.h"
''',
                        sources=[path.join(marxscr, *f) for f in sources],
                        include_dirs=[path.join(marxscr, *f) for f in headers],
                        libraries=['marx', 'pfile', 'jdmath', 'jdfits'],
                        library_dirs=[marxlib]
                        )

if __name__ == "__main__":
    ffi.compile()
