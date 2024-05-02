MARXS
=====

Multi-Architecture-Raytrace-Xraymission-Simulator

.. image:: https://img.shields.io/pypi/v/marxs
   :alt: PyPI - Version

.. image:: https://zenodo.org/badge/34215358.svg
   :target: https://zenodo.org/badge/latestdoi/34215358
   :alt: Link to last release with DOI

.. image:: https://github.com/Chandra-MARX/marxs/actions/workflows/ci_tests.yml/badge.svg
   :target: https://github.com/Chandra-MARX/marxs/actions/
   :alt: Continuous integration status

.. image:: https://readthedocs.org/projects/marxs/badge/?version=latest
   :target: http://marxs.readthedocs.io/en/latest/
   :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/Chandra-MARX/marxs/badge.svg?branch=master 
   :target: https://coveralls.io/github/Chandra-MARX/marxs?branch=master
   :alt: Fraction of code covered by tests

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
   :target: http://www.astropy.org/
   :alt: Powered by astropy

What does MARXS do?
-------------------

MARXS (Multi-Architecture-Raytrace-Xraymission-Simulator) is a toolsuite to simulate
X-ray observatories. It is primarily aimed at astronomical X-ray satellites and sounding rocket
payloads, but can be used to ray-trace experiments in the laboratory as well.
MARXS performs polarization Monte-Carlo ray-trace simulations from a source (astronomical or lab) through a collection of
optical elements such as mirrors, baffles, and gratings to a detector.

MARXS modular structure is designed to serve two main use cases:

- **Build-your-own instrument**:
  Instrument designers can construct any X-ray experiment from a set of building
  blocks such as mirrors, diffraction gratings and CCD detectors.
- **Simulate science observations**:
  Given an instrument configuration, simulate the detector output for any set of X-ray
  sources in the lab or on the sky.


Installation
------------
See the [installation instructions](https://marxs.readthedocs.io/en/latest/installation.html)
in the documentation.

Citation
--------
Look at the [CITATION](CITATION) file for the citation information.

Contributions
-------------
We welcome contributions through [pull requests on Github](https://github.com/Chandra-MARX/marxs/pulls).
Please follow our [code of conduct](CODE_OF_CONDUCT.md).



