import numpy as np
from .params import cosmo_params
from .cosmo import rho_cz

Msol_cgs = 1.989e33
rhocrit =  1.87847e-29 * cosmo_params['hh']**2
G_cgs = 6.67259e-8 #cm3/g/s2
kpc_cgs = 3.086e21
fb = cosmo_params['Omega_b']/cosmo_params['Omega_m']

def r200(M, z):
    '''radius of a sphere with density 200 times the critical density of the universe.
    Input mass in solar masses. Output radius in cm.
    '''
    M_cgs = M*Msol_cgs
    om = cosmo_params['Omega_m']
    ol = cosmo_params['Omega_L']
    Ez2 = om * (1 + z)**3 + ol
    ans = (3 * M_cgs / (4 * np.pi * 200.*rhocrit*Ez2))**(1.0/3.0)
    return ans


def rho_gnfw1h(xx, M, z, theta):
    '''generalized NFW profile describing the gas density [g/cm3],
    parameters have mass and redshift dependence, as in Battaglia 2016
    '''
    #rho0 = 4e3 * (M/1e14)**0.29 * (1+z)**(-0.66)
    #xc = 0.5
    al = 0.88 * (M/1e14)**(-0.03) * (1+z)**0.19
    #bt = 3.83 * (M/1e14)**0.04 * (1+z)**(-0.025)
    gm = -0.2
    rho0,xc,bt = theta
    #rho0,al,bt = theta
    ans = 10**rho0 * (xx/xc)**gm / ((1 + (xx/xc)**al)**((bt-gm)/al))
    ans *= rho_cz(z) * fb
    return ans

def rho_gnfw2h(xx, a2):
    rho = np.genfromtxt('data/rhoGNFW_M2e+13_z0.57.txt')
    rat = rho[:,0]
    rho2h = rho[:,2]
    ans = a2 * np.interp(xx,rat,rho2h)
    return ans

def rho_gnfw(xx, theta):
    theta1h = theta[0], theta[1], theta[2]
    theta2h = theta[3]
    ans = rho_gnfw1h(xx,M,z,theta1h) + rho_gnfw2h(xx, theta2h)
    return ans


def Pth_gnfw1h(x,M,z,theta):
    '''generalized NFW profile describing the thermal pressure [cgs units],
    parameters have mass and redshift dependence, as in Battaglia et al. (2012)
    '''
    if isinstance(M, float) is True:
        r200c = r200(M,z)
        rvir=r200(M,z)/kpc_cgs/1e3 #Mpc
        x /= rvir
        M_cgs = M*Msol_cgs
        P200c = G_cgs * M_cgs* 200. * rho_cz(z) * fb /(2.*r200c)
        #P0, xc, bt = theta
        P0, al, bt = theta
        #P0 = 18.1 * (m/1e14)**0.154 * (1+z)**(-0.758)
        #al = 1.0
        #bt = 4.35 * (m/1e14)**0.0393 * (1+z)**0.415
        xc = 0.497 * (M/1e14)**(-0.00865) * (1+z)**0.731
        gm = -0.3
        ans = P0 * (x/xc)**gm * (1+(x/xc)**al)**(-bt)
        ans *= P200c
        return ans
    else:
        logm = np.log10(M)
        h,b_edges = np.histogram(logm, bins=49)
        b_cen = np.array([(b_edges[i]+b_edges[i-1])*0.5 for i in range(1,len(b_edges))])
        b_len = np.array([(b_edges[i]-b_edges[i-1]) for i in range(1,len(b_edges))])
        integ = np.sum(10**b_len*h)
        p = h/integ
        pth = []
        for i in range(0, len(b_cen)):
            m = 10**b_cen[i]
            r200c = r200(m,z)
            rvir=r200(m,z)/kpc_cgs/1e3 #Mpc
            M_cgs = m*Msol_cgs
            P200c = G_cgs * M_cgs* 200. * rho_cz(z) * fb /(2.*r200c)
            P0, al, bt = theta
            xc = 0.497 * (m/1e14)**(-0.00865) * (1+z)**0.731
            gm = -0.3
            xx = x/rvir
            pth.append(P0 * (xx/xc)**gm * (1+(xx/xc)**al)**(-bt) * P200c)   
        pth = np.array(pth)
        pth_av = np.average(pth, weights=p, axis=0)
        return pth_av

def Pth_gnfw2h(xx):
    pth = np.genfromtxt('data/PthGNFW_M2e+13_z0.57.txt')
    rat = pth[:,0]
    pth2h = pth[:,2]
    ans = np.interp(xx,rat,pth2h)
    return ans

def Pth_gnfw(xx, theta):
    ans = Pth_gnfw1h(xx,M,z,theta) + Pth_gnfw2h(xx)
    return ans
