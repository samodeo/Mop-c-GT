from headers import *

import cosmo
reload(cosmo)
from cosmo import *

import gnfw
reload(gnfw)
from gnfw import *

import obb
reload(obb)
from obb import *

'''
The OBB model of rho and Pth is obtained through 'make_a_obs_profile()'.

The GNFW model of rho is obtained through 'make_a_obs_profile_sim_rho()'.

The GNFW model of Pth is obtained through 'make_a_obs_profile_sim_pth()'.
'''


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


def project_prof_beam_sim_rho(tht,Mvir,theta_rho):
    theta_sim_rho = theta_rho

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

    thta_smooth = (np.arange(NNR2) + 1.)*dtht
    thta = thta[:,None]
    thta2_smooth = (np.arange(NNR2) + 1.)*dtht2
    thta2 = thta2[:,None]

    rho2D_beam  = np.trapz(thta_smooth  * rho2D  * np.exp(-0.5*thta_smooth**2 /sigmaBeam**2)
                                 * special.iv(0, thta_smooth *thta / sigmaBeam**2), x=thta_smooth, axis=1)
    rho2D2_beam = np.trapz(thta2_smooth * rho2D2 * np.exp(-0.5*thta2_smooth**2/sigmaBeam**2)
                                 * special.iv(0, thta2_smooth*thta2/ sigmaBeam**2), x=thta2_smooth,axis=1)

    thta = (np.arange(NNR) + 1.)*dtht
    thta2 = (np.arange(NNR) + 1.)*dtht2

    area_fac = 2.0*np.pi*dtht*np.sum(thta)

    rho2D_beam  *= np.exp(-0.5*thta**2 /sigmaBeam**2) / sigmaBeam**2
    rho2D2_beam *= np.exp(-0.5*thta2**2/sigmaBeam**2) / sigmaBeam**2

    sig  = 2.0*np.pi*dtht *np.sum(thta *rho2D_beam)
    sig2 = 2.0*np.pi*dtht2*np.sum(thta2*rho2D2_beam)

    v_rms = 7.e-4 # v_rms/c
    sig_all_beam = (2*sig - sig2) * v_rms * ST_CGS * TCMB * 1e6 * ((2. + 2.*XH)/(3.+5.*XH)) / MP_CGS #/ (np.pi * np.radians(tht/60.)**2)

    return sig_all_beam

def project_prof_beam_sim_pth(tht,Mvir,theta_pth):
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

    Pth2D = 2*np.trapz(Pth_gnfw1h(rint/rvir,theta_sim_pth,Mvir), x=rad * kpc_cgs, axis=1) * 1e3
    Pth2D2 = 2*np.trapz(Pth_gnfw1h(rint2/rvir,theta_sim_pth,Mvir), x=rad2 * kpc_cgs, axis=1) * 1e3

    thta_smooth = (np.arange(NNR2) + 1.)*dtht
    thta = thta[:,None]
    thta2_smooth = (np.arange(NNR2) + 1.)*dtht2
    thta2 = thta2[:,None]

    Pth2D_beam  = np.trapz(thta_smooth  * Pth2D  * np.exp(-0.5*thta_smooth**2 /sigmaBeam**2)
                                 * special.iv(0, thta_smooth *thta / sigmaBeam**2), x=thta_smooth, axis=1)
    Pth2D2_beam = np.trapz(thta2_smooth * Pth2D2 * np.exp(-0.5*thta2_smooth**2/sigmaBeam**2)
                                 * special.iv(0, thta2_smooth*thta2/ sigmaBeam**2), x=thta2_smooth, axis=1)


    thta = (np.arange(NNR) + 1.)*dtht
    thta2 = (np.arange(NNR) + 1.)*dtht2

    area_fac = 2.0*np.pi*dtht*np.sum(thta)

    Pth2D_beam  *= np.exp(-0.5*thta**2 /sigmaBeam**2) / sigmaBeam**2
    Pth2D2_beam *= np.exp(-0.5*thta2**2/sigmaBeam**2) / sigmaBeam**2

    sig_p  = 2.0*np.pi*dtht*np.sum(thta*Pth2D_beam)
    sig2_p = 2.0*np.pi*dtht2*np.sum(thta2*Pth2D2_beam)

    sig_all_p_beam = (2*sig_p - sig2_p) * ST_CGS/(ME_CGS*C_CGS**2) * TCMB * 1e6 * ((2. + 2.*XH)/(3.+5.*XH)) #/ area_fac # muK

    return sig_all_p_beam


def find_params_M(M,theta_0):
    M_use = M*Msol_cgs
    theta0 = np.append(M_use,theta_0)
    beta_0 = 1.1
    con_test = con(M_use)
    theta2 = np.array([beta_0 ,con_test*1.01])
    ans = return_prof_pars(theta2,theta0)
    return ans


def make_a_obs_profile(thta_arc,M,zz,theta_0):
    thta2 = find_params_M(M,theta_0)
    M_use = M*Msol_cgs
    rho = np.zeros(len(thta_arc))
    pth = np.zeros(len(thta_arc))
    for ii in range(len(thta_arc)):
        temp = project_prof_beam(thta_arc[ii],M_use,theta_0,thta2)
        rho[ii] = temp[0]
        pth[ii] = temp[1]
    return rho,pth

def make_a_obs_profile_sim_rho(thta_arc,M,zz,theta_rho):
    M_use = M*Msol_cgs
    rho = np.zeros(len(thta_arc))
    for ii in range(len(thta_arc)):
        temp = project_prof_beam_sim_rho(thta_arc[ii],M_use,theta_rho)
        rho[ii] = temp
    return rho

def make_a_obs_profile_sim_pth(thta_arc,M,zz,theta_pth):
    M_use = M*Msol_cgs
    pth = np.zeros(len(thta_arc))
    for ii in range(len(thta_arc)):
        temp = project_prof_beam_sim_rho(thta_arc[ii],M_use,theta_pth)
        pth[ii] = temp
    return pth
