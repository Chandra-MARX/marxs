MARXS
=====

Multi-Architecture-Raytrace-Xraymission-Simulator

[![PyPI - Version](https://img.shields.io/pypi/v/marxs)](https://pypi.org/project/marxs/)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4255147.svg)](https://doi.org/10.5281/zenodo.4255147)

[![CI tests](https://github.com/Chandra-MARX/marxs/actions/workflows/ci_tests.yml/badge.svg)](https://github.com/Chandra-MARX/marxs/actions/)

[![Documentation Status](https://readthedocs.org/projects/marxs/badge/?version=latest)](http://marxs.readthedocs.io/en/latest/)

[![Coverage Status](https://coveralls.io/repos/github/Chandra-MARX/marxs/badge.svg?branch=master)](https://coveralls.io/github/Chandra-MARX/marxs?branch=master)

[![Powered by Astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)

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



