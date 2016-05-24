************
Installation
************

Requirements
============

- numpy >=1.8
- scipy

.. _sect-installmarxccode:

`classic marx`_ C code
======================
The `classic marx`_ code is an optional dependency. By default, it is not used and all
modules build on `classic marx`_ will be unavailable.

In order to build the interface to the `classic marx`_ C code, you need to set the path
to both, the `classic marx`_ source code and an installed version of `classic marx`_ on your
machine in the ``setup.cfg`` file in the root directory of the installation
package.

These are the steps required to use the interface to `classic marx`_:

- Edit ``setup.cfg`` with the path to the `classic marx`_ source code and to compiled binaries.
- Once that is done, install marxs with ``python setup.py install`` in the root directory of the MARXS distribution.

In the current (`classic marx`_ 5.1) default setup, `classic marx`_ compiles static libraries, not
shared objects. Static libraries are a tiny bit better in performance at the
cost of extra difficulty of linking them into shared objects. Since `classic marx`_ is
not meant to be used a library for external functions (like this python
module), the default installation settings are tuned for performance.
On some architectures (tested on 32-bit Ubuntu GNU/Linux) linking the static
libraries works, on other you might see an error like this::

    relocation R_X86_64_32 against `.text' can not be used when making a shared object; recompile with -fPIC

In that case, simply recompile and install `classic marx`_ as *P* osition *I* ndependent
*C* ode. In the `classic marx`_ source code directory:: 

    make distclean
    ./configure --prefix=/path/to/your/instalation/ CFLAGS="-O2 -g -fPIC"
    make clean
    make
    make install

I promise that the performance difference is so small, you won't notice
it when you run the `classic marx`_ version, but it allows the setup process of
this python module to compile the interface to use those libraries.

If you are a developer, you might want to tell git to ignore the local path
that you put into ``setup.cfg`` to avoid committing and pushing that to the
repro accidentially::

  git update-index --assume-unchanged setup.cfg

(However, note that this will make some manual fixing necessary if the upstream
``setup.cfg`` changes, e.g. because we decide to add a new option. See 
``git help update-index`` for more explanation.)
