************
Installation
************

Requirements
============

MARXS has few hard requirements:

- MARXS requires Python version > 3.6.
- `numpy <http://www.numpy.org/>`_
- `astropy`_
- `transforms3d <https://matthew-brett.github.io/transforms3d/>`_

Numpy and astropy are best installed with a package manager such as conda. See the `astropy installation instructions for a detailed discussion <https://astropy.readthedocs.io/en/stable/install.html>`_. ``transforms3d`` is easily installed with::

.. code-block:: bash

    pip install transforms3d

The following Python packages are strongly recommended, but most parts of MARXS will work without them:

- `scipy <http://www.numpy.org/>`_
- matplotlib
- `mayavi <https://docs.enthought.com/mayavi/mayavi/>`_ (for 3 D output)
- jsonschema
- pyyaml

Again, all but mayavi are available through common package managers such as
conda, ``apt-get`` etc. For `mayavi
<https://docs.enthought.com/mayavi/mayavi/>`_ see `the mayavi installation
instructions <https://docs.enthought.com/mayavi/mayavi/installation.html#installing-with-pip>`_.
  
In addition MARXS has an interface to the `classic marx`_ C code used to simulate the Chandra mirrors (:ref:`sect-installmarxccode`).
  
Install the python code
=======================

setup.py
--------

To download the latest development version of MARXS:

.. code-block:: bash

   $ git clone https://github.com/Chandra-MARXS/marxs.git
   $ cd marxs

Now you install, run tests or build the documentation:

.. code-block:: bash

   $ python setup.py install
   $ python setup.py test
   $ python setup.py build_docs

If you want to contribute to MARXS, but are not familiar with Python or
git or Astropy yet, please have a look at the  
`Astropy developer documentation <http://docs.astropy.org/en/latest/#developer-documentation>`__.

  
.. _sect-installmarxccode:

`classic marx`_ C code
======================
The `classic marx`_ code is an optional dependency. By default, it is not used and all
modules build on `classic marx`_ will be unavailable.

In order to build the interface to the `classic marx`_ C code, you need to set the path
to the `classic marx`_ source code *and* an installed version of `classic marx`_ on your
machine in the ``setup.cfg`` file in the root directory of the installation
package *before* you call ``python setup.py install`` in the root directory of the MARXS distribution.

The current `classic marx`_ default setup compiles static libraries, not
shared objects. Static libraries are a tiny bit better in performance at the
cost of extra difficulty of linking them into shared objects. Since `classic marx`_ is
not meant to be used a library for external functions (like this python
module), the default installation settings are tuned for performance.
On some architectures (tested on 32-bit Ubuntu GNU/Linux) linking the static
libraries works, on other you might see an error like this: ``relocation R_X86_64_32 against `.text' can not be used when making a shared object; recompile with -fPIC``.
In that case, simply recompile and install `classic marx`_ as position independent
code (PIC). In the `classic marx`_ source code directory:: 

.. code-block:: bash

    make distclean
    ./configure --prefix=/path/to/your/instalation/ CFLAGS="-O2 -g -fPIC"
    make clean
    make
    make install

I promise that the performance difference is so small, you won't notice
it when you run the `classic marx`_ version, but it allows the setup process of
this python module to compile the interface to use those libraries.
