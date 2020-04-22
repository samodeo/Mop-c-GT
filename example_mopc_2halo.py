import time
import numpy as np
import mopc as mop
import emcee
from scipy.interpolate import Rbf

thta_arc = np.linspace(1., 6., 9) #arcmin
M = 2e13
#M = np.array(np.genfromtxt('data/mass_distrib.txt'))
Mmed = np.median(M)
z = 0.57
beam = 1.41 #arcmin

def read_rho2h():
    zbin = ['0.45', '0.48', '0.52', '0.56', '0.6', '0.64', '0.68'] 
    x1 = np.logspace(np.log10(0.1),np.log10(10),50)
    rho2h = []
    for ix in range(0,len(x1)):
        rho2h_z = []
        for iz in zbin:
            rho = np.genfromtxt('data/rhoGNFW_M2e+13'+'_z'+iz+'.txt')        
            rho2h_z.append(rho[ix,2])
        rho2h.append(rho2h_z) 
    rho2h = np.array(rho2h)
    return rho2h
   
rho_grid = read_rho2h() 

def rho_grid_rbf():
    zbin = np.array([0.45, 0.48, 0.52, 0.56, 0.6, 0.64, 0.68])
    x1 = np.logspace(np.log10(0.005),np.log10(5),50) #Mpc
    zarr = rho_grid * 0.0 + zbin[None,:]
    xarr = rho_grid * 0.0 + x1[:,None]
    rbf = Rbf(xarr, zarr, rho_grid, function='linear') 
    return rbf

rho2h = rho_grid_rbf() 


'''Battaglia 2016'''
rho0_0 = 4e3 * (Mmed/1e14)**0.29 * (1+z)**(-0.66)
xc_0 = 0.5
al_0 = 0.88 * (Mmed/1e14)**(-0.03) * (1+z)**0.19
bt_0 = 3.83 * (Mmed/1e14)**0.04 * (1+z)**(-0.025)
a2_0 = 1.

par = [np.log10(rho0_0), xc_0, bt_0, a2_0]

s = time.time()
ksz_model = mop.make_a_obs_profile_sim_rho(thta_arc,M,z,par,beam,rho2h)
e = time.time()

print(e-s)
print()
print(ksz_model)