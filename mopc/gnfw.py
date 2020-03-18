from headers import *

def r200(M, z_input=None):
    '''radius of a sphere with density 200 times the critical density of the universe,
    input mass in cgs 
    '''
    if z_input is not None:
        z = z_input
    else:
        z = Params['z']
    om = Params['Omega_m']
    ol = Params['Omega_L']
    Ez2 = om * (1 + z)**3 + ol
    ans = (3 * M / (4 * np.pi * 200.*rhocrit*Ez2))**(1.0/3.0)
    return ans


def rho_gnfw1h(xx, theta):
    '''generalized NFW profile describing the gas density [g/cm3],
    parameters have mass and redshift dependence, as in Battaglia 2016
    '''
    M = Params['M']
    z = Params['z']
    fb = Params['Omega_b']/Params['Omega_m']
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
    rat = rho[:,1]
    rho2h = rho[:,3]
    ans = a2 * np.interp(xx,rat,rho2h)
    return ans

def rho_gnfw(xx, theta):
    theta1h = theta[0], theta[1], theta[2]
    theta2h = theta[3]
    ans = rho_gnfw1h(xx, theta1h) + rho_gnfw2h(xx, theta2h)
    return ans


def Pth_gnfw1h(x,theta, Mcgs):
    '''generalized NFW profile describing the thermal pressure [cgs units],
    parameters have mass and redshift dependence, as in Battaglia et al. (2012)
    '''
    G_cgs = 6.67259e-8 #cm3/g/s2
    z = Params['z']
    fb = Params['Omega_b']/Params['Omega_m']
    M = M_cgs / Msol_cgs
    r200c = r200(M_cgs,z)
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

def Pth_gnfw2h(xx):
    pth = np.genfromtxt('data/PthGNFW_M2e+13_z0.57.txt')
    rat = pth[:,1]
    pth2h = pth[:,3]
    ans = np.interp(xx,rat,pth2h)
    return ans

def Pth_gnfw(xx, theta):
    ans = Pth_gnfw1h(xx, theta, M) + Pth_gnfw2h(xx)
    return ans
