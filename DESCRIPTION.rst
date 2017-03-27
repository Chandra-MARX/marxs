MARXS (Multi-Architecture-Raytrace-Xraymission-Simulator) is a toolsuite to simulate
X-ray observatories. It is primarily aimed at astronomical X-ray satellites and sounding rocket
payloads, but can be used to ray-trace experiments in the laboratory as well.
MARXS performs polarization Monte-Carlo ray-trace simulations from a source (astronomical or lab) through a collection of
optical elements such as mirrors, baffles, and gratings to a detector.

MARXS modular structure is designed to serve two main use cases:

- **Build-your-own instrument**:
  Instrument designers can construct any X-ray experiement from a set of building
  blocks such as mirrors, diffraction gratings and CCD detectors.
- **Simulate science observations**:
  Given an instrument configuration, simulate the detector output for any set of X-ray
  sources in the lab or on the sky.
