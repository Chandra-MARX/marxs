output_cols currently are all float. Mechanisms ot use other types (e.g. int for grating order). On the other hand, how do I mask invalid values there? Masked Table (performance?)? For now, I have np.nan...

Write in docs that geometry is not to be changed after init, because init
derives other quantities from it (e.g. the normal), which would not be updated.
Maybe also override __setitem__ to make unaccessible after init phase
(or redesign so that derived numbersare updated dynamically, e.g. use property
and derive on the fly - however, I fear a performance hit)

For Skycoords (source, pointing: Use astropy coordinates?)
  would give easy lookup for realistic coords. 
  Might not be too important if SIMPUT becomes standard input format.


Define everything in cgs system (angles in deg) or use astropy for units?
For now, the Marx mirror is the only think with hardcoded units.
(mm and keV). For now, make that standard units for everything. If I ever want to change that I need to deal with unit conversions.

also check out np.linalg and scipy.linalg
remember np.empty

Here are some helpers for matix making etc.
http://matthew-brett.github.io/transforms3d/index.html


Some functions (e.g. intersect_line_plane) are not testest for special cases
(what happens if the line is in the plane? Or parallel to the plane?)
Unlikely that that comes up in practice, but I definitely should add checks,
add tests and deal with it in the code.

Prune p=0 photons automatically in every step or keep them in with p=0?

So far only marx_mirror actually destroys photons.

Maybe the (to be written) logic in simulator.py can wrap process_photons and pass only photons with probability > 0. That would save time (considerably if the mirror is not that efficient) and avoid errors doe to photon paths the are not physically possible (since a p=0 photons still has a dir, it might turn up at impossible locations).
