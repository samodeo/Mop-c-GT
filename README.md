# Mop-c GT
## Model-to-observable projection code for galaxy thermodynamics

 Authors: Stefania Amodeo, Nicholas Battaglia
 
Requirements: numpy, scipy.

 Takes in input models of density and pressure radial profiles of halos of given mass and redshift.

 Returns 2D profiles of observable quantities, specifically the temperature shifts in CMB maps measured within disks of varying radii due to the kSZ and tSZ effects, respectively, by implementing:

 - line-of-sight projection
 - instrumental beam convolution
 - aperture photometry filter

 Applications to:
 - generalized NFW parametric models
 - polytropic gas model based on Ostriker, Bode, Balbus (2005).
 
Includes implementation of the analytical model of the “two-halo term” (contribution to the halo gas profiles from neighboring halos), following the halo model of Vikram et al. (2017) (based on the formalism of Cooray & Sheth 2002)
Extra packages needed for the two-halo computation: 
- hmf by Murray, Power and Robotham (2013) : https://github.com/steven-murray/hmf 
- colossus by Diemer 2018: https://bdiemer.bitbucket.io/colossus/installation.html

Presented in Amodeo et al. in prep.
