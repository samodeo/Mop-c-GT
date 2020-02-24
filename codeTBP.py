import numpy as np
from scipy.special import spence
from scipy.optimize import fmin
from scipy.optimize import newton
from scipy.integrate import quad
from scipy import special

Params = {'Omega_m':0.25,'hh':0.7,'Omega_L':0.75,'Omega_b':0.044,'rhoc_0':2.77525e2,'C_OVER_HUBBLE':2997.9,'z':0.57}
fb = Params['Omega_b']/Params['Omega_m']
Gmax = 0.216216538797
Gravity = 6.67259e-8
rhocrit =  1.87847e-29 * Params['hh']**2
Msol_cgs = 1.989e33
kpc_cgs = 3.086e21
kev_erg = 1.60218e-9
C_CGS = 2.99792e+10
ST_CGS = 6.65246e-25
ME_CGS = 9.10939E-28
MP_CGS = 1.6726219e-24
TCMB   = 2.725
GMAX = 0.2162
XH = 0.76
delx = 0.01


def hub_func(z):
    '''Hubble function
    '''
    Om = Params['Omega_m']
    Ol = Params['Omega_L']
    O_tot = Om + Ol
    ans = np.sqrt(Om*(1.0 + z)**3 + Ol + (1 - O_tot)*(1 + z)**2)
    return ans

def rho_cz(z):
    '''critical density in cgs
    '''
    Ez2 = Params['Omega_m']*(1+z)**3. + (1-Params['Omega_m'])
    return rhocrit * Ez2


def ComInt(z):
    '''comoving distance integrand
    '''
    ans = 1.0/hub_func(z)
    return ans


def ComDist(z):
    '''comoving distance
    '''
    Om = Params['Omega_m']
    Ol = Params['Omega_L']
    O_tot = Om + Ol
    Dh = Params['C_OVER_HUBBLE']/Params['hh']
    ans = Dh*quad(ComInt,0,z)[0]
    if (O_tot < 1.0): ans = Dh / np.sqrt(1.0-O_tot) *  np.sin(np.sqrt(1.0-O_tot) * quad(ComInt,0,z)[0])
    if (O_tot > 1.0): ans = Dh / np.sqrt(O_tot-1.0) *  np.sinh(np.sqrt(O_tot-1.0) * quad(ComInt,0,z)[0])
    return ans

def AngDist(z):
    '''angular distance
    '''
    ans = ComDist(z)/(1.0+z)
    return ans

def nfw(x):
    '''shape of a NFW profile (NFW 1997, ApJ,490, 493)
    '''
    ans = 1./(x*(1 + x)**2)
    return ans

def gx(x):
    ans = (np.log(1. + x) - x/(1. + x))
    return ans

def gc(c):
    ans = 1./(np.log(1. + c) - c/(1. + c))
    return ans

def Hc(c):
    ans = (-1.*np.log(1 + c)/(1. + c) + c*(1. + 0.5*c)/((1. + c)**2))/gx(c)
    return ans

def Sc(c):
    ans = (0.5*np.pi**2 - np.log(c)/2. - 0.5/c - 0.5/(1 + c)**2 - 3/(1 + c) +
           np.log(1 + c)*(0.5+0.5/c**2-2/c-1/(1+c)) +
           1.5*(np.log(1 + c))**2 + 3.*spence(c+1))

    return ans

def del_s(c):
    ans = Sc(c) / (Sc(c) + (1./c**3)*Hc(c)*gx(c))
    return ans

def K_c(c): #without GMAX
    ans = 1./3.* Hc(c)/(1.-del_s(c))
    return ans


def sig_dm2(x,c): 
    '''EQ 14 Lokas & Mamon 2001
    '''
    ans = 0.5*x*c*gc(c)*(1 + x)**2 *(np.pi**2 - np.log(x) - (1./x)
                                     - (1./(1. + x)**2) - (6./(1. + x))
                                     + np.log(1. + x)*(1. + (1./x**2) - 4./x - 2/(1 + x))
                                    + 3.*(np.log(1. + x))**2 + 6.*spence(x+1))
    return ans


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

def con(Mvir, z_input=None):
    '''
    concentration parameter from Duffy et al. (2008)
    input mass in cgs
    '''
    M = Mvir / Msol_cgs
    if z_input is not None:
        z = z_input
    else:
        z = Params['z']
    ans = 5.71 / (1 + z)**(0.47) * (M / 2e12)**(-0.084)
    return ans

def rho_dm(x,Mvir):
    '''NFW profile describing the dark matter density [g/cm3]
    '''
    c = con(Mvir)
    rvir = r200(Mvir)
    ans = Mvir*(c/rvir)**3 / (4.*np.pi*gx(c)) * nfw(x)
    return ans

def jx(x,c):
    ans = 1. - np.log(1. + x)/x
    ind = np.where(x > c) #[0]
    if (len(ind) > 0):
        ans[ind] = 1. -1./(1. + c) - (np.log(1. + c) - c/(1.+c))/x[ind]
    return ans

def jx_f(x,c):
    if (x <= c):
        ans = 1. - np.log(1. + x)/x
    else:
        ans = 1. -1./(1. + c) - (np.log(1. + c) - c/(1.+c))/x
    return ans

def fx (x,c):
    ans = np.log(1. + x)/x - 1./(1. + c)
    ind = np.where(x > c)[0]
    if (len(ind) > 0):
        ans = (np.log(1. + c)/c - 1./(1. + c))*c/x
    return ans

def fstar_func(Mvir):
    '''Giodini 2009, modified by 0.5
    '''
    ans = 2.5e-2 * (Mvir / (7e13*Msol_cgs))**(-0.37) 
    return ans

def xs_min_func(x,Mvir):
    c = con(Mvir)
    fstar = fstar_func(Mvir)
    ans = gx(c)*fstar/(1. + fstar) - gx(x)
    return ans

def xs_func(Mvir):
    x0 = 1.0
    xs = newton(xs_min_func, x0, args=(Mvir,))
    return xs

def Ks(x_s,Mvir):
    c = con(Mvir)
    xx = np.arange(delx/2.,x_s,delx)
    ans = 1./gx(c)*(np.sum(Sc(xx)*xx**2) - 2./3.*np.sum(fx(xx,c)*xx/(1. + xx)**2) )*delx
    return ans

def n_exp(gamma):
    '''exponent of the polytopic e.o.s.
    '''
    ans = 1. / (gamma - 1)
    return ans

def theta_func(x,Mvir,theta,theta2):
    '''polytropic variable
    '''
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    c = con(Mvir)
    rvir = r200(Mvir)
    nn = n_exp(gamma)
    #print(jx(x,c).shape)
    ans = (1. - beta*jx(x,c)/(1. + nn))
    return ans

def theta_func_f(x,Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    c = con(Mvir)
    rvir = r200(Mvir)
    nn = n_exp(gamma)
    ans = (1. - beta*jx_f(x,c)/(1. + nn))
    return ans

def rho_use(x,Mvir,theta, theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    ans = (theta_func(x,Mvir,theta,theta2))**nn
    return ans

def rho(x,Mvir,theta, theta2):
    gamma,alpha,Ef = theta
    P_0, rho_0, x_f = theta2
    nn = n_exp(gamma)
    c = con(Mvir)
    rvir = r200(Mvir)
    beta = rho_0/P_0 * Gravity*Mvir/rvir*c/gx(c)
    theta2_use = beta, x_f
    #print(theta_func(x,Mvir,theta,theta2_use))
    ans = rho_0*(theta_func(x,Mvir,theta,theta2_use))**nn
    return ans


def rho_outtest(x,Mvir,theta, theta2):
    gamma,alpha,Ef = theta
    P_0, rho_0, x_f = theta2
    nn = n_exp(gamma)
    c = con(Mvir)
    rvir = r200(Mvir)
    beta = rho_0/P_0 * Gravity*Mvir/rvir*c/gx(c)
    theta2_use = beta, x_f
    #print "inside rhoout", theta2_use
    ans = rho_0*(theta_func(x,Mvir,theta,theta2_use))**nn
    return ans

def Pnth_th(x,Mvir,theta):
    gamma,alpha,Ef = theta
    c = con(Mvir)
    ans = 1. - alpha*(x/c)**0.8
    return ans

def Pth(x,Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    P_0, rho_0, x_f = theta2
    c = con(Mvir)
    rvir = r200(Mvir)
    nn = n_exp(gamma)
    beta = rho_0/P_0 * Gravity*Mvir/rvir*c/gx(c)
    theta2_use = beta, x_f
    ans = P_0*(theta_func(x,Mvir,theta,theta2_use))**(nn+1.) * Pnth_th(x,Mvir,theta)
    return ans

def Pth_use(x,Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    ans = (theta_func(x,Mvir,theta,theta2))**(nn+1.) * Pnth_th(x,Mvir,theta)
    return ans

def Ptot(Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    P_0, rho_0, x_f = theta2
    nn = n_exp(gamma)
    rvir = r200(Mvir)
    c = con(Mvir)
    beta = rho_0/P_0 * Gravity*Mvir/rvir*c/gx(c)
    theta2_use = beta, x_f
    ans = P_0*(theta_func_f(x_f,Mvir,theta,theta2_use))**(nn+1.)
    return ans

def Ptot_use(Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    ans = (theta_func_f(x_f,Mvir,theta,theta2))**(nn+1.)
    return ans

def Pnth(x,Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    P_0, rho_0, x_f = theta2
    c = con(Mvir)
    nn = n_exp(gamma)
    rvir = r200(Mvir)
    beta = rho_0/P_0 * Gravity*Mvir/rvir*c/gx(c)
    theta2_use = beta, x_f
    ans = alpha*(x/c)**0.8 * P_0*(theta_func(x,Mvir,theta,theta2_use))**(nn+1.)
    return ans

def Pnth_use(x,Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    c = con(Mvir)
    nn = n_exp(gamma)
    ans = alpha*(x/c)**0.8 * (theta_func(x,Mvir,theta,theta2))**(nn+1.)
    return ans

def I2_int(Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    c = con(Mvir)
    xx = np.arange(delx/2.,x_f,delx)
    ans = np.sum(fx(xx,c)*rho_use(xx,Mvir,theta,theta2)*xx**2)*delx
    return ans

def I3_int(Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    xx = np.arange(delx/2.,x_f,delx)
    ans = np.sum(Pth_use(xx,Mvir,theta,theta2) *xx**2)*delx
    return ans

def I4_int(Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    xx = np.arange(delx/2.,x_f,delx)
    ans = np.sum(Pnth_use(xx,Mvir,theta,theta2)*xx**2)*delx
    return ans

def L_int(Mvir,theta,theta2):
    gamma,alpha,Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    xx = np.arange(delx/2.,x_f,delx)
    ans = np.sum(rho_use(xx,Mvir,theta,theta2)*xx**2)*delx
    return ans

def rho_0_func(theta0,theta2):
    Mvir,gamma,alpha,Ef = theta0
    theta = [gamma,alpha,Ef]
    c = con(Mvir)
    rvir = r200(Mvir)
    fstar = fstar_func(Mvir)
    ans = Mvir*(fb-fstar) / (4.*np.pi * L_int(Mvir,theta,theta2)*(rvir/c)**3)
    return ans

def P_0_func(theta0,theta2,rho_0):
    Mvir,gamma,alpha,Ef = theta0
    beta, x_f = theta2
    c = con(Mvir)
    rvir = r200(Mvir)
    ans = rho_0/beta * Gravity*Mvir/rvir*c/gx(c)
    return ans

def findroots2(theta2,theta0):
    Mvir,gamma,alpha,Ef = theta0
    theta = [gamma,alpha,Ef]
    beta, x_f = theta2
    c = con(Mvir)
    rvir = r200(Mvir)
    x_s = xs_func(Mvir)
    fstar = fstar_func(Mvir)

    E_inj = Ef * gx(c) * rvir * fstar / (Gravity*Mvir*c) * C_CGS**2

    Eq1 = (3./2.*(1. + fstar) * (K_c(c)*(3.-4.*del_s(c)) + Ks(x_s,Mvir))  - E_inj + 1./3.* (1.+fstar) *Sc(c) / gx(c) * (x_f**3 - c**3)
           - I2_int(Mvir,theta,theta2)/L_int(Mvir,theta,theta2)
           + 3./2. * I3_int(Mvir,theta,theta2)/(beta*L_int(Mvir,theta,theta2))
           + 3.* I4_int(Mvir,theta,theta2)/(beta*L_int(Mvir,theta,theta2)))

    Eq2 = (1.+fstar)*Sc(c) / gx(c) * (beta*L_int(Mvir,theta,theta2)) - Ptot_use(Mvir,theta,theta2)

    ans = Eq1**2 + Eq2**2
    return ans

def return_prof_pars(theta2,theta0):
    Mvir,gamma,alpha,Ef = theta0
    beta, x_f = theta2
    ans = fmin(findroots2, theta2, args=(theta0,),disp=False)
    beta_ans, x_f_ans = ans
    rho_0 = rho_0_func(theta0,ans)
    P_0 = P_0_func(theta0,ans,rho_0)
    return P_0, rho_0, x_f_ans

def findroots(theta2,theta0):
    Mvir,gamma,alpha,Ef = theta0
    theta = [gamma,alpha,Ef]
    beta, x_f = theta2
    c = con(Mvir)
    rvir = r200(Mvir)
    E_inj = Ef * gx(c) * rvir / (Gravity*Mvir*c) * C_CGS**2

    Eq1 = (3./2. * (K_c(c)*(3.-4.*del_s(c))) - E_inj + 1./3.* Sc(c) / gx(c) * (x_f**3 - c**3)
           - I2_int(Mvir,theta,theta2)/L_int(Mvir,theta,theta2)
           + 3./2. * I3_int(Mvir,theta,theta2)/(beta*L_int(Mvir,theta,theta2))
           + 3.* I4_int(Mvir,theta,theta2)/(beta*L_int(Mvir,theta,theta2)))
    Eq2 = Sc(c) / gx(c) * (beta*L_int(Mvir,theta,theta2)) - Ptot_use(Mvir,theta,theta2)
    return (Eq1,Eq2)


def rho_sim(theta, x):
    a1,a2,a3 = theta
    gamma = 0.2
    ans = 10**a1 / ((x/0.5)**gamma * (1 + (x/0.5)**a2)**((a3 - gamma)/a2))
    ans *=  Params['rhoc_0'] * Msol_cgs / kpc_cgs / kpc_cgs / kpc_cgs * Params['Omega_b']/Params['Omega_m']
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
    rho = np.genfromtxt('/Users/samodeo/Desktop/2halo_GNFW_CMASS/wf/rhoGNFW_M2e+13_z0.57.txt')
    rat = rho[:,1]
    rho2h = rho[:,3]
    ans = a2 * np.interp(xx,rat,rho2h)
    return ans

def rho_gnfw(xx, theta):
    theta1h = theta[0], theta[1], theta[2]
    theta2h = theta[3]
    ans = rho_gnfw1h(xx, theta1h) + rho_gnfw2h(xx, theta2h)
    return ans

def Pth_sim(x,theta):
    G_cgs = 6.67259e-8 #cm3/g/s2
    M = Params['M']
    z = Params['z']
    M_cgs = M * Msol_cgs
    r200c = r200(M_cgs,z)
    P0,xc,bt = theta
    al = 1.0
    gm = 0.3
    ans = P0 / ((x*xc)**gm * (1 + (x*xc)**al)**((bt-gm)/al))
    ans *= G_cgs * M_cgs* 200. * rho_cz(z) * fb /(2.*r200c)
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
    pth = np.genfromtxt('/Users/samodeo/Desktop/2halo_GNFW_CMASS/wf/PthGNFW_M2e+13_z0.57.txt')
    rat = pth[:,1]
    pth2h = pth[:,3]
    ans = np.interp(xx,rat,pth2h)
    return ans

def Pth_gnfw(xx, theta):
    ans = Pth_gnfw1h(xx, theta, M) + Pth_gnfw2h(xx)
    return ans

def project_prof_beam(tht,Mvir,theta,theta2):

    disc_fac = np.sqrt(2)
    l0 = 30000.
    NNR = 100
    NNR2 = 2*NNR

    fwhm = 1.4
    fwhm *= np.pi / (180.*60.)
    sigmaBeam = fwhm / np.sqrt(8.*np.log(2.))

    XH = 0.76
    z = Params['z']
    P0, rho0, x_f = theta2
    fstar = fstar_func(Mvir)

    AngDis = AngDist(z)

    rvir = r200(Mvir)/kpc_cgs/1e3
    c = con(Mvir)

    r_ext = AngDis*np.arctan(np.radians(tht/60.))
    r_ext2 = AngDis*np.arctan(np.radians(tht*disc_fac/60.))

    rvir_arcmin = 180.*60./np.pi * np.tan(rvir/AngDis) #arcmin
    rvir_ext = AngDis*np.arctan(np.radians(rvir_arcmin/60.))
    rvir_ext2 = AngDis*np.arctan(np.radians(rvir_arcmin*disc_fac/60.))

    rad = np.logspace(-3, 1, 1e3) #in MPC
    rad2 = np.logspace(-3, 1, 1e3) #in MPC

    radlim = r_ext
    radlim2 = r_ext2

    dtht = np.arctan(radlim/AngDis)/NNR # rads
    dtht2 = np.arctan(radlim2/AngDis)/NNR # rads

    thta = (np.arange(NNR) + 1.)*dtht
    thta2 = (np.arange(NNR) + 1.)*dtht2

    thta_smooth = (np.arange(NNR2) + 1.)*dtht
    thta2_smooth = (np.arange(NNR2) + 1.)*dtht2

    thta_smooth = thta_smooth[:,None]
    thta2_smooth = thta2_smooth[:,None]

    rint  = np.sqrt(rad**2  + thta_smooth**2 *AngDis**2)
    rint2  = np.sqrt(rad2**2  + thta2_smooth**2 *AngDis**2)

    rho2D = 2*np.trapz(rho(rint/rvir*c,Mvir,theta,theta2), x=rad * kpc_cgs, axis=1) * 1e3 #/((2*np.trapz(rho(rnorm/rvir*c,Mvir,theta,theta2), x=rad * kpc_cgs) * 1e3) *(np.sum(thta_norm)*2.0*np.pi*dtht_norm))
    rho2D2 = 2*np.trapz(rho(rint2/rvir*c,Mvir,theta,theta2), x=rad2 * kpc_cgs, axis=1) * 1e3 #/ ((2*np.trapz(rho(rnorm/rvir*c,Mvir,theta,theta2), x=rad * kpc_cgs) * 1e3)* (np.sum(thta_norm)*2.0*np.pi*dtht_norm))

    Pth2D = 2*np.trapz(Pth(rint /rvir*c,Mvir,theta,theta2), x=rad * kpc_cgs, axis=1) * 1e3
    Pth2D2 = 2*np.trapz(Pth(rint2 /rvir*c,Mvir,theta,theta2), x=rad2 * kpc_cgs, axis=1) * 1e3

    thta_smooth = (np.arange(NNR2) + 1.)*dtht
    thta = thta[:,None]
    thta2_smooth = (np.arange(NNR2) + 1.)*dtht2
    thta2 = thta2[:,None]

    rho2D_beam  = np.trapz(thta_smooth  * rho2D  * np.exp(-0.5*thta_smooth**2 /sigmaBeam**2)
                                 * special.iv(0, thta_smooth *thta / sigmaBeam**2), x=thta_smooth, axis=1)
    rho2D2_beam = np.trapz(thta2_smooth * rho2D2 * np.exp(-0.5*thta2_smooth**2/sigmaBeam**2)
                                 * special.iv(0, thta2_smooth*thta2/ sigmaBeam**2), x=thta2_smooth,axis=1)


    Pth2D_beam  = np.trapz(thta_smooth  * Pth2D  * np.exp(-0.5*thta_smooth**2 /sigmaBeam**2)
                                 * special.iv(0, thta_smooth *thta / sigmaBeam**2), x=thta_smooth, axis=1)
    Pth2D2_beam = np.trapz(thta2_smooth * Pth2D2 * np.exp(-0.5*thta2_smooth**2/sigmaBeam**2)
                                 * special.iv(0, thta2_smooth*thta2/ sigmaBeam**2), x=thta2_smooth, axis=1)

    thta = (np.arange(NNR) + 1.)*dtht
    thta2 = (np.arange(NNR) + 1.)*dtht2

    area_fac = 2.0*np.pi*dtht*np.sum(thta)

    rho2D_beam  *= np.exp(-0.5*thta**2 /sigmaBeam**2) / sigmaBeam**2
    rho2D2_beam *= np.exp(-0.5*thta2**2/sigmaBeam**2) / sigmaBeam**2
    Pth2D_beam  *= np.exp(-0.5*thta**2 /sigmaBeam**2) / sigmaBeam**2
    Pth2D2_beam *= np.exp(-0.5*thta2**2/sigmaBeam**2) / sigmaBeam**2

    sig  = 2.0*np.pi*dtht *np.sum(thta *rho2D_beam)
    sig2 = 2.0*np.pi*dtht2*np.sum(thta2*rho2D2_beam)

    v_rms = 7.e-4 #1e-3 #v_rms/c
    sig_all_beam = (2*sig - sig2) * v_rms * ST_CGS * TCMB * 1e6 * ((2. + 2.*XH)/(3.+5.*XH)) / MP_CGS

    sig_p  = 2.0*np.pi*dtht*np.sum(thta*Pth2D_beam)
    sig2_p = 2.0*np.pi*dtht2*np.sum(thta2*Pth2D2_beam)

    sig_all_p_beam = (2*sig_p - sig2_p) * ST_CGS/(ME_CGS*C_CGS**2) * TCMB * 1e6 * ((2. + 2.*XH)/(3.+5.*XH))#/ area_fac # muK

    return sig_all_beam, sig_all_p_beam


def project_prof_beam_sim(tht,Mvir,theta_pth):
    #theta_sim_rho = theta_rho
    theta_sim_rho = np.array([3.6337402156859753, 1.0369351928324118, 3.3290812595973063])
    #theta_sim_pth = np.array([18.1, 1.0, 4.35])
    theta_sim_pth = theta_pth

    disc_fac = np.sqrt(2)
    l0 = 30000.
    NNR = 100
    NNR2 = 2.*NNR

    fwhm = 1.4
    fwhm *= np.pi / (180.*60.)
    sigmaBeam = fwhm / np.sqrt(8.*np.log(2.))

    drint = 1e-3 * (kpc_cgs * 1e3)
    XH = 0.76
    z = Params['z']

    AngDis = AngDist(z)

    rvir = r200(Mvir)/kpc_cgs/1e3 #Mpc
    c = con(Mvir)

    r_ext = AngDis*np.arctan(np.radians(tht/60.))
    r_ext2 = AngDis*np.arctan(np.radians(tht*disc_fac/60.))

    rvir_arcmin = 180.*60./np.pi * np.tan(rvir/AngDis) #arcmin
    rvir_ext = AngDis*np.arctan(np.radians(2*rvir_arcmin/60.))

    rad = np.logspace(-3, 1, 1e3) #Mpc
    rad2 = np.logspace(-3, 1, 1e3)

    radlim = r_ext
    radlim2 = r_ext2

    dtht = np.arctan(radlim/AngDis)/NNR # rads
    dtht2 = np.arctan(radlim2/AngDis)/NNR # rads

    thta = (np.arange(NNR) + 1.)*dtht
    thta2 = (np.arange(NNR) + 1.)*dtht2

    thta_smooth = (np.arange(NNR2) + 1.)*dtht
    thta2_smooth = (np.arange(NNR2) + 1.)*dtht2

    thta_smooth = thta_smooth[:,None]
    thta2_smooth = thta2_smooth[:,None]

    rint  = np.sqrt(rad**2 + thta_smooth**2 *AngDis**2)
    rint2  = np.sqrt(rad2**2 + thta2_smooth**2 *AngDis**2)

    rho2D = 2*np.trapz(rho_gnfw1h(rint/rvir,theta_sim_rho), x=rad * kpc_cgs, axis=1) * 1e3
    rho2D2 = 2*np.trapz(rho_gnfw1h(rint2/rvir,theta_sim_rho), x=rad2 * kpc_cgs, axis=1) * 1e3

    Pth2D = 2*np.trapz(Pth_gnfw1h(rint/rvir,theta_sim_pth,Mvir), x=rad * kpc_cgs, axis=1) * 1e3
    Pth2D2 = 2*np.trapz(Pth_gnfw1h(rint2/rvir,theta_sim_pth,Mvir), x=rad2 * kpc_cgs, axis=1) * 1e3

    thta_smooth = (np.arange(NNR2) + 1.)*dtht
    thta = thta[:,None]
    thta2_smooth = (np.arange(NNR2) + 1.)*dtht2
    thta2 = thta2[:,None]

    rho2D_beam  = np.trapz(thta_smooth  * rho2D  * np.exp(-0.5*thta_smooth**2 /sigmaBeam**2)
                                 * special.iv(0, thta_smooth *thta / sigmaBeam**2), x=thta_smooth, axis=1)
    rho2D2_beam = np.trapz(thta2_smooth * rho2D2 * np.exp(-0.5*thta2_smooth**2/sigmaBeam**2)
                                 * special.iv(0, thta2_smooth*thta2/ sigmaBeam**2), x=thta2_smooth,axis=1)


    Pth2D_beam  = np.trapz(thta_smooth  * Pth2D  * np.exp(-0.5*thta_smooth**2 /sigmaBeam**2)
                                 * special.iv(0, thta_smooth *thta / sigmaBeam**2), x=thta_smooth, axis=1)
    Pth2D2_beam = np.trapz(thta2_smooth * Pth2D2 * np.exp(-0.5*thta2_smooth**2/sigmaBeam**2)
                                 * special.iv(0, thta2_smooth*thta2/ sigmaBeam**2), x=thta2_smooth, axis=1)


    thta = (np.arange(NNR) + 1.)*dtht
    thta2 = (np.arange(NNR) + 1.)*dtht2

    area_fac = 2.0*np.pi*dtht*np.sum(thta)

    rho2D_beam  *= np.exp(-0.5*thta**2 /sigmaBeam**2) / sigmaBeam**2
    rho2D2_beam *= np.exp(-0.5*thta2**2/sigmaBeam**2) / sigmaBeam**2
    Pth2D_beam  *= np.exp(-0.5*thta**2 /sigmaBeam**2) / sigmaBeam**2
    Pth2D2_beam *= np.exp(-0.5*thta2**2/sigmaBeam**2) / sigmaBeam**2


    sig  = 2.0*np.pi*dtht *np.sum(thta *rho2D_beam)
    sig2 = 2.0*np.pi*dtht2*np.sum(thta2*rho2D2_beam)

    v_rms = 7.e-4 # v_rms/c
    sig_all_beam = (2*sig - sig2) * v_rms * ST_CGS * TCMB * 1e6 * ((2. + 2.*XH)/(3.+5.*XH)) / MP_CGS #/ (np.pi * np.radians(tht/60.)**2)

    sig_p  = 2.0*np.pi*dtht*np.sum(thta*Pth2D_beam)
    sig2_p = 2.0*np.pi*dtht2*np.sum(thta2*Pth2D2_beam)

    sig_all_p_beam = (2*sig_p - sig2_p) * ST_CGS/(ME_CGS*C_CGS**2) * TCMB * 1e6 * ((2. + 2.*XH)/(3.+5.*XH)) #/ area_fac # muK

    return sig_all_beam, sig_all_p_beam


def find_params_M(M,theta_0):
    M_use = M*Msol_cgs
    theta0 = np.append(M_use,theta_0)
    beta_0 = 1.1
    con_test = con(M_use)
    theta2 = np.array([beta_0 ,con_test*1.01])
    ans = return_prof_pars(theta2,theta0)
    return ans


def make_a_obs_profile(thta_arc,M,zz,theta_0):
    Params['z'] = zz
    thta2 = find_params_M(M,theta_0)
    M_use = M*Msol_cgs
    rho = np.zeros(len(thta_arc))
    pth = np.zeros(len(thta_arc))
    for ii in range(len(thta_arc)):
        temp = project_prof_beam(thta_arc[ii],M_use,theta_0,thta2)
        rho[ii] = temp[0]
        pth[ii] = temp[1]
    return rho,pth

def make_a_obs_profile_sim(thta_arc,M,zz,theta_1):
    Params['z'] = zz
    M_use = M*Msol_cgs
    rho = np.zeros(len(thta_arc))
    pth = np.zeros(len(thta_arc))
    for ii in range(len(thta_arc)):
        temp = project_prof_beam_sim(thta_arc[ii],M_use,theta_1)
        rho[ii] = temp[0]
        pth[ii] = temp[1]
    return rho,pth
