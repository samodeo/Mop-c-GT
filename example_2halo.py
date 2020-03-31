import numpy as np
import matplotlib.pyplot as plt
import mopc
from mopc.params import cosmo_params
from mopc.cosmo import rho_cz
from mopc.gnfw import r200
from mopc import two_halo as two

'''choose halo redshift and mass [Msun]
'''
z = 0.57
m = 2e13

Msol_cgs = 1.989e33
G_cgs = 6.67259e-8 #cm3/g/s2
fb = cosmo_params['Omega_b']/cosmo_params['Omega_m']

print()
print('**Computing 3D profiles of density and preassure in cgs units**')
print()
print('starting with density')

x = np.logspace(np.log10(0.1),np.log10(10),50)

rho1h = two.rho_gnfw(x,m,z)  #g/cm3

rho2h=[]
for ix in range(len(x)):
    rho2h.append(two.rho_2h(x[ix],m,z))
rho2h = np.array(rho2h)  #g/cm3


rhogas_c = fb * rho_cz(z) #g/cm3

print('save rho table and plot')
header_rho='R/R200c, rho1h, rho2h, (rho1h+rho2h) \nrho in [g/cm3]'
np.savetxt('data/rhoGNFW_M{:.0e}_z{}.txt'.format(m,z), np.transpose([x,rho1h,rho2h,rho1h+rho2h]),newline='\n', header=header_rho)

plt.figure(1)
plt.loglog(x,rho1h/rhogas_c,'--',label='1-halo')
plt.loglog(x,rho2h/rhogas_c,'--',label='2-halo')
plt.loglog(x,(rho1h+rho2h)/rhogas_c)
plt.xlabel(r'$r \, / \, R_{200c}$')
plt.ylabel(r"$\rho_{gas}(r) / \rho_{c}$ ")
plt.title(r"$M = {:.0e} M_\odot$, z={}".format(m,z))
plt.legend()
plt.savefig('fig/rhoGNFW_M'+'%1.0e' %m+'_z'+str(z)+'.pdf')

print('done with density')
print()
print('starting with pressure')


Pth1h = two.Pth_gnfw(x,m,z) #[g/cm/s2]

Pth2h=[]
for ix in range(len(x)):
    Pth2h.append(two.Pth_2h(x[ix],m,z))
Pth2h = np.array(Pth2h) 

print('save Pth table and plot')
header='R/R200c, Pth1h, Pth2h, (Pth1h+Pth2h) \nPth in [g cm^-1 s^-2]'
np.savetxt('data/PthGNFW_M{:.0e}_z{}.txt'.format(m,z), np.transpose([x,Pth1h,Pth2h,Pth1h+Pth2h]),newline='\n', header=header)

P200c = G_cgs * m*Msol_cgs * 200. * rho_cz(z) * fb /(2.*r200(m,z)) 

plt.figure(2)
plt.loglog(x,Pth1h/P200c,'--',label='1-halo')
plt.loglog(x,Pth2h/P200c,'--',label='2-halo')
plt.loglog(x,(Pth1h+Pth2h)/P200c)
plt.xlabel(r'$r \, / \, R_{200c}$')
plt.ylabel(r"$P_{th}(r) / P_{200}$ ")
plt.title(r"$M = {:.0e} M_\odot$, z={}".format(m,z))
plt.legend()
plt.savefig('fig/PthGNFW_M'+'%1.0e' %m+'_z'+str(z)+'.pdf')

print('done with pressure')
print('done')
