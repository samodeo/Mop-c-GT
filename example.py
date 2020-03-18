import numpy as np
import matplotlib.pyplot as plt
import Mop-c-GT as mop

'''set halo redshift and mass [Msun]
'''
z = 0.57
m = 2e13

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
temp_ksz_gnfw = mop.make_a_obs_profile_sim_rho(theta,m,z,par_rho)


##########################################################################################
'''project a gNFW profile of the thermal Pressure [cgs unit] from Battaglia et al. 2012
into a T_tSZ profile [muK arcmin^2]
'''
P0 = 18.1 * (m/1e14)**0.154 * (1+z)**(-0.758)
al = 1.0
bt = 4.35 * (m/1e14)**0.0393 * (1+z)**0.415
par_pth = [P0, al,bt]
temp_tsz_gnfw = mop.make_a_obs_profile_sim_pth(theta,m,z,par_pth)


##########################################################################################
'''project density and thermal pressure models [cgs unit] from Ostriker, Bode, Balbus (2005)
into T_kSZ and T_tSZ profiles [muK arcmin^2]
'''
gamma = 1.2 #polytropic index
alpha_NT = 0.13 #non-thermal pressure norm.
eff = 2e-5 #feedback efficiency
par_obb = [gamma,alpha_NT,eff] 
temp_ksz_obb = mop.make_a_obs_profile(theta,m,z,par_obb)[0]
temp_tsz_obb = mop.make_a_obs_profile(theta,m,z,par_obb)[1]
