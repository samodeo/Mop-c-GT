from headers import *

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
    #print("inside rhoout", theta2_use)
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