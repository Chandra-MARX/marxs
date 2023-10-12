README
======

The ARF files in this directory can be used to account for the effects
of insufficient order sorting. There were created with the following
order-sorting strategy:

$description

The order sorting window (the range of CCD energies that are assigned to
one order) is finite and does not cover the full probability distribution
of detected photon energies for that order. Thus, some photons are not
extracted and the effective area in an order is smaller.
Also, photons from neighboring orders may leak into a main order, e.g.
photons from order -1 or -3 might be assigned to order -2. In this directory
the following offsets from the main order are considered:

$offsetorders

How do I use these files?
-------------------------

Each PHA spectrum will have more than one ARF and RMF file. Since the
RMF files do not depend on the order sorting, they are not stored in
this directory, but in the parent directory.

For example, to fit (or fake!) a spectrum for order -4, you need to load
the following files:

$filenamemain
$filenameupper
$filenamelower

and their associated RMFs from the parent directory.

The exact command to load and use more than one ARF and RMF per dataset depends
on the program you are using, e.g. XSPEC, ISIS, Sherpa, etc.
The arcus-simulation group is preparing instructions for different programs.

So, to simulate all orders that Arcus will see, you need to create
a whole bunch of spectra with 2-3 ARFs and RMFs each. Note that some orders
only have 2 ARFs, because order 1 is not contaminated by order 0 and the
simulations just stop at some maximum order (in reality, there will be more
orders, but with so little signal that we probably won't see them above the
background).


What is "near" and "far"?
-------------------------

Arcus has two cameras of 8 CCDs each. We currently split ARF and RMF
files by camera. File called "near" fall on the camera that also
contains the 0th order, files called "far" on the other. Some orders
can be seen on both cameras and thus have both a "near" and a "far"
file. In the simulation, treat those as independent, i.e. simulate two
spectra for order -1, one of them with the "near" files and one with
the "far" files.

Why are the arfs here and the other files in the parent directory?
------------------------------------------------------------------
ARFs fall into three groups:

- far: The RMFs is independent of order-sorting and the same RMF
  file can be used for every simulation. Thus, we keep the RMF files in the
  root directory.

- Order 0: spatially distinct from the dispersed orders and thus no
  order sorting is necessary. Because the order 0 ARF is independent of the
  order sorting, it is found in the parent directory.

- near: The orders get close together for larger values for m lambda.
  Order-sorting is not a problem close to the 0th order, because all
  orders are well separated.
  Thus, the "near" files are the same for every simulation and are
  kept in the parent directory.



Version
-------

ARF and RMF files carry version information such as the date and the
program version used to write them in their fits header. Check there
if files have been manually edited. However, for simplicity, some
information is reproduced below in plain text.

$tagversion
