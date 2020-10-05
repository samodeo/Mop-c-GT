import numpy as np
import matplotlib.pyplot as plt
import mopc as mop


'''choose halo redshift and mass [Msun]
'''
z = 0.57
m = 2e13

'''choose radial range (x = r/rvir)
'''
x = np.logspace(np.log10(0.1),np.log10(10),100) 

'''choose angular range [eg. arcmin]
'''
theta = np.arange(100)*0.05 + 0.5 
theta = np.linspace(0.5,5.5,50)
sr2sqarcmin = 3282.8 * 60.**2

'''
choose observing frequency [GHz]
'''
nu = 150.

'''
read beam profile
'''
theta_rad, beam = np.loadtxt('data/beam_example.txt',unpack=True)
def f_beam(tht):
    tht_in = theta_rad
    return np.interp(tht, tht_in, beam, period=np.pi)


##########################################################################################
'''project a gNFW density profile [cgs unit] from Battaglia 2016
into a T_kSZ profile [muK arcmin^2]
'''
rho0 = np.log10(4e3 * (m/1e14)**0.29 * (1+z)**(-0.66))
xc = 0.5
bt = 3.83 * (m/1e14)**0.04 * (1+z)**(-0.025)
a2h= 1.1
par_rho = [rho0,xc,bt,a2h]
rho_gnfw = mop.rho_gnfw1h(x,m,z,par_rho[:3])
temp_ksz_gnfw = mop.make_a_obs_profile_sim_rho(theta,m,z,par_rho,f_beam)

plt.figure(0,figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.loglog(x, rho_gnfw, lw=3, label='gNFW density')
plt.xlabel(r'x',size=14)
plt.ylabel(r"$\rho_{gas}(x) [g/cm^3]$ ",size=14)
plt.subplot(1, 2, 2)
plt.plot(theta, temp_ksz_gnfw*sr2sqarcmin,lw=3)
plt.xlabel(r'$\theta$ [arcmin]',size=14)
plt.ylabel(r'$T_{kSZ} [\mu K \cdot arcmin^2]$',size=14)
plt.tight_layout()
plt.savefig('fig/rho_gnfw.pdf')

##########################################################################################
'''project a gNFW profile of the thermal Pressure [cgs unit] from Battaglia et al. 2012
into a T_tSZ profile [muK arcmin^2]
''' 

P0 = 18.1 * (m/1e14)**0.154 * (1+z)**(-0.758)
al = 1.
bt = 4.35 * (m/1e14)**0.0393 * (1+z)**0.415
a2h = 0.7
par_pth = [P0, al,bt,a2h]
pth_gnfw = mop.Pth_gnfw1h(x,m,z,par_pth[:3])
temp_tsz_gnfw = mop.make_a_obs_profile_sim_pth(theta,m,z,par_pth,nu,f_beam)

plt.figure(1,figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.loglog(x, pth_gnfw,lw=3, label='gNFW pressure')
plt.xlabel(r'x',size=14)
plt.ylabel(r"$P_{th}(x) [dyne/cm^2]$",size=14)
plt.subplot(1, 2, 2)
plt.plot(theta, temp_tsz_gnfw*sr2sqarcmin,lw=3)
plt.xlabel(r'$\theta$ [arcmin]',size=14)
plt.ylabel(r'$T_{tSZ} [\mu K \cdot arcmin^2]$',size=14)
plt.tight_layout()
plt.savefig('fig/pth_gnfw.pdf')
