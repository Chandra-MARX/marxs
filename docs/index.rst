Multi-Architecture-Raytrace-Xraymission-Simulator (MARXS)
=========================================================

.. include:: ../DESCRIPTION.rst

The documentation for MARXS is split into several parts. After installing MARXS it first
explains how to use MARXS to simulate an (astronomical) observation for an existing
instrument configuration, e.g. from the examples distributed with MARX or provided by
an instrument team. The following section is aimed at instrument teams and proposers.

.. note:: 

   MARXS has been used to simulate several instruments (e.g. ARCUS and Lynx), but is still under
   rapid develoment. We will try to mimimize API changes from now on, but will implement those
   as necessary (e.g. the astropy units system will be used on more places in the future to
   reduce unit mismatches). The :ref:`changelog` will always contain a full list of changes
   between released versions.

**Install MARXS**

.. toctree::
   :maxdepth: 2

   install

**Using MARXS to simulate an observation**

.. toctree::
   :maxdepth: 2

   runexample
   source
   results
   missions

**Define an instrument or mission with MARXS**

.. toctree::
   :maxdepth: 2
	       
   conventions
   optics
   design
   newopticalelements
   visualization
   examples
   utils

**Contribute to MARXS development**

.. toctree::
   :maxdepth: 2
	      
   contributing

**MARXS project details**

.. toctree::
   :maxdepth: 2

   support
   changelog
   credits


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

