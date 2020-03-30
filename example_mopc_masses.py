import numpy as np
import matplotlib.pyplot as plt
import mopc as mop


'''Start with a population of halos with a mass distribution, at a certain redshift.
'''
masses = np.genfromtxt('data/mass_distrib.txt')
z = 0.5

'''choose radial range (x = r/rvir)
'''
x = np.logspace(np.log10(0.1),np.log10(10),100) 

'''
Compute the average Pressure profile [cgs unit] of the halo sample, 
from Battaglia et al. 2012, with each halo 
'''
m_med = np.median(masses)
P0 = 18.1 * (m_med/1e14)**0.154 * (1+z)**(-0.758)
al = 1.0
bt = 4.35 * (m_med/1e14)**0.0393 * (1+z)**0.415
par_pth = [P0, al,bt]
pth_av = mop.Pth_gnfw1h(x,masses,z,par_pth)


############
#PROJECTION#
############

'''choose angular range [eg. arcmin]
'''
theta = np.arange(100)*0.05 + 0.5 
sr2sqarcmin = 3282.8 * 60.**2

'''
choose observing frequency [GHz]
'''
nu = 150.

'''project a gNFW profile of the thermal Pressure [cgs unit] from Battaglia et al. 2012
into a T_tSZ profile [muK arcmin^2]
''' 
temp_tsz_gnfw = mop.make_a_obs_profile_sim_pth(theta,masses,z,par_pth,nu)

plt.clf()
plt.figure(1,figsize=(11, 4))

plt.subplot(1, 2, 1)
plt.loglog(x, pth_av,lw=3, label='gNFW pressure')
plt.xlabel(r'x',size=14)
plt.ylabel(r"$P_{th}(x) [dyne/cm^2]$",size=14)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(theta, temp_tsz_gnfw*sr2sqarcmin,lw=3)
plt.xlabel(r'$\theta$ [arcmin]',size=14)
plt.ylabel(r'$T_{tSZ} [\mu K \cdot arcmin^2]$',size=14)
plt.tight_layout()
plt.savefig('/fig/pth_gnfw_mw.pdf')
