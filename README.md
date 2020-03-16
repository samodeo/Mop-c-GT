# Mop-c GT
 Model-to-observable projection code for galaxy thermodynamics
 
 Authors: Stefania Amodeo, Nicholas Battaglia

 Takes in input models of density and pressure radial profiles of halos of given mass and redshift.
 
 Returns 2D profiles of observable quantities, specifically the temperature shifts in CMB maps measured within disks of varying radii due to the kSZ and tSZ effects, respectively, by implementing:
 
 - line-of-sight projection
 - instrumental beam convolution
 - aperture photometry filter
 
 Applications to:
 - generalized NFW parametric models
 - polytropic gas model based on Ostriker, Bode, Balbus (2005). 

Only requires numpy and scipy.

Presented in Amodeo et al. in prep.