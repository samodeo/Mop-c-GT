import numpy as np
import matplotlib.pyplot as plt
import Mop-c-GT as mop

'''set halo redshift and mass [Msun]
'''
z = 0.57
m = 2e13

'''choose radial range (x = r/rvir)
'''
x = np.logspace(np.log10(0.01),np.log10(10),100) 

'''choose angular range [eg. arcmin]
'''
theta = np.arange(100)*0.05 + 0.5 


##########################################################################################
'''project a gNFW density profile [cgs unit] from Battaglia 2016
into a T_kSZ profile [muK arcmin^2]
'''
rho0 = 4e3 * (m/1e14)**0.29 * (1+z)**(-0.66)
xc = 0.5
bt = 3.83 * (m/1e14)**0.04 * (1+z)**(-0.025)
par_rho = [rho0,xc,bt]
rho_gnfw = mop.rho_gnfw1h(x,par_rho)
temp_ksz_gnfw = mop.make_a_obs_profile_sim_rho(theta,m,z,par_rho)

plt.figure(0)
plt.subplot(1, 2, 1)
plt.loglog(x, rho_gnfw, label='gNFW density')
plt.xlabel(r'x')
plt.ylabel(r"$\rho_{gas}(r) [g/cm^3]$ ")
plt.subplot(1, 2, 2)
plt.plot(theta, temp_ksz_gnfw)
plt.xlabel(r'$\theta$ [arcmin]')
plt.ylabel(r'$T_{tSZ} [\mu K \cdot arcmin^2]$')
plt.legend()
plt.show()

##########################################################################################
'''project a gNFW profile of the thermal Pressure [cgs unit] from Battaglia et al. 2012
into a T_tSZ profile [muK arcmin^2]
''' 

P0 = 18.1 * (m/1e14)**0.154 * (1+z)**(-0.758)
al = 1.0
bt = 4.35 * (m/1e14)**0.0393 * (1+z)**0.415
par_pth = [P0, al,bt]
pth_gnfw = mop.Pth_gnfw1h(x,par_pth,m)
temp_tsz_gnfw = mop.make_a_obs_profile_sim_pth(theta,m,z,par_pth)

plt.figure(1)
plt.subplot(1, 2, 1)
plt.loglog(x, pth_gnfw, label='gNFW pressure')
plt.xlabel(r'x')
plt.ylabel(r"$P_{th}(r) [dyne/cm^2]$")
plt.subplot(1, 2, 2)
plt.plot(theta, temp_tsz_gnfw)
plt.xlabel(r'$\theta$ [arcmin]')
plt.ylabel(r'$T_{tSZ} [\mu K \cdot arcmin^2]$')
plt.legend()
plt.show()

##########################################################################################
'''project density and thermal pressure models [cgs unit] from Ostriker, Bode, Balbus (2005)
into T_kSZ and T_tSZ profiles [muK arcmin^2]
'''
gamma = 1.2 #polytropic index
alpha_NT = 0.13 #non-thermal pressure norm.
eff = 2e-5 #feedback efficiency
par_obb = [gamma,alpha_NT,eff] 
par2_obb = mop.find_params_M(M,par_obb) #P_0, rho_0, x_f
rho_obb = mop.rho(x,m,par_obb,par2_obb) 
pth_obb = mop.Pth(x,m,par_obb,par2_obb)
temp_ksz_obb = mop.make_a_obs_profile(theta,m,z,par_obb)[0]
temp_tsz_obb = mop.make_a_obs_profile(theta,m,z,par_obb)[1]

plt.figure(2)
plt.subplot(1, 2, 1)
plt.loglog(x, rho_obb, label='OBB density')
plt.xlabel(r'x')
plt.ylabel(r"$\rho_{gas}(r) [g/cm^3]$ ")
plt.subplot(1, 2, 2)
plt.plot(theta, temp_ksz_obb)
plt.xlabel(r'$\theta$ [arcmin]')
plt.ylabel(r'$T_{tSZ} [\mu K \cdot arcmin^2]$')
plt.legend()
plt.show()

plt.figure(3)
plt.subplot(1, 2, 1)
plt.loglog(x, pth_obb, label='OBB pressure')
plt.xlabel(r'x')
plt.ylabel(r"$P_{th}(r) [dyne/cm^2]$")
plt.subplot(1, 2, 2)
plt.plot(theta, temp_tsz_obb)
plt.xlabel(r'$\theta$ [arcmin]')
plt.ylabel(r'$T_{tSZ} [\mu K \cdot arcmin^2]$')
plt.legend()
plt.show()
