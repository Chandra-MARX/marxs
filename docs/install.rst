============
Installation
============

`marx`_ C code
==============
In order to build the interface to the `marx`_ C code, you need to set the path
to both, the `marx`_ source code and an installed version of `marx`_ on your
machine in the ``setup.cfg`` file in the root directory of the installation
package.

If you are a developer, you might want to tell git to ignore the local path
that you put into ``setup.cfg`` to avoid committing and pushing that to the
repro accidentially::

  git update-index --assume-unchanged setup.cfg

(However, note that this will make some manual fixing necessary if the upstream
``setup.cfg`` changes, e.g. because we decide to add a new option. See 
``git help update-index`` for more explanation.)

.. todo::
   Implement a mechanism where `marx`_ is downloaded and installed in the
   installation process to a known location, possibly to extern in this
   package?
   (E.g. make it a git submodule?) and bootstrap it into the installation process?

In the current (`marx`_ 5.1) default setup, the `marx`_ compiles static libraries, not
shared objects. Static libraries are a tiny bit better in performance at the
cost of extra difficulty of linking them into shared objects. Since `marx`_ is
not meant to be used a library for external functions (like this python
module), the default installations settings are tuned for performance.
On some architectures (tested on 32-bit Ubuntu GNU/Linux) linking the static
libraries works, on other you might see an error like this::

    relocation R_X86_64_32 against `.text' can not be used when making a shared object; recompile with -fPIC

In that case, simply recompile and install `marx`_ as *P* osition *I* ndependent
*C* ode. In the `marx`_ source code directory:: 

    make distclean
    ./configure --prefix=/melkor/d1/guenther/marx/dev CFLAGS="-O0 -g -fPIC"
    make
    make install

I promise that the performance difference is so small, you won't notice
it when you run the classic `marx`_ version, but it allows the setup process of
this python module to compile the interface to use those libraries.
