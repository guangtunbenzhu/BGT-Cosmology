"""
Power Spectra
Everything in Comoving frame
"""

# TO-DO: 
# For consistency, in halofit by Smith+2003, change all parameters to physical unit (without h)
# Add z to be84 (sigma_R)
# nonlinear PS: halofit, coyote
# Be careful with slicing and assignment: Python creates a "view" (or a "copy")

import cosmology as cosmo
import numpy as np
from scipy import integrate, interpolate

# Doesnot include Eisenstein & Hu 1997 model for high baryon density (no cosmological constant) yet
# Adapted from Eisenstein & Hu 1999 power.c
# TFmdm_onek_mpc
# CosPar = {'Omega_M': Matter, 'Omega_b': Baryon, 'Omega_nu': Massive neutrinos, 'n_degen_nu': Number of degenerate massive neutrino species,
#           'Omega_L', Cosmological constant, 'h': Hubble constant}

# Note kk should be Mpc^-1 but not h Mpc^-1
# Return (T(k)*D(z)/D0)^2*k**ns; What's the unit???
def EH_TF_mpc(kk, z, CosPar):

    print "IF YOU SEE ME TOO MANY TIMES, YOU SHOULD VECTORIZE YOUR CODE. MAYBE YOU SHOULD VECTORIZE ME!"
    #print "Initializing..."
    ## Set Parameters

    # Check for strange inputs
    if CosPar['Omega_b']<0.: raise ValueError("Omega_b < 0.. Can't continue.")
    if CosPar['Omega_nu']<0.: raise ValueError("Omega_nu < 0.. Can't continue.")
    if CosPar['h']<0.: raise ValueError("h < 0. Can't continue.")
    if CosPar['h']>2.: raise ValueError("h > 2. Can't continue.")
    if CosPar['n_degen_nu']<1.: raise ValueError("n_degen_nu < 1. Can't continue.")
    if z<0.: raise ValueError("z < 0. Can't continue.")
    if z>99.: raise ValueError("z > 99. Can't continue.")

    Omega_M = CosPar['Omega_M']
    Omega_b = CosPar['Omega_b']
    Omega_nu = CosPar['Omega_nu']
    n_degen_hdm = CosPar['n_degen_nu']
    Omega_L = CosPar['Omega_L']
    h = CosPar['h']
    ns = CosPar['ns']

    Omhh = Omega_M*h**2
    Obhh = Omega_b*h**2
    Onhh = Omega_nu*h**2
    f_baryon = Omega_b/Omega_M
    f_hdm = Omega_nu/Omega_M
    f_cdm = 1.-f_baryon-f_hdm
    f_cb = f_cdm+f_baryon
    f_bnu = f_baryon+f_hdm

    # Compute the equality scale
    z_equality = cosmo.z_equality(CosPar)
    k_equality = cosmo.k_equality(CosPar)
    sound_horizon_fit = cosmo.sound_horizon(CosPar)
    #print "z_eq=",z_equality
    #print "k_eq=",k_equality
    #print "Sound Horizon: ", sound_horizon_fit, " Mpc"

    # Compute the drag epoch and sound horizon
    z_drag_b1 = 0.313*pow(Omhh, -0.419)*(1.+0.607*pow(Omhh, 0.674))
    z_drag_b2 = 0.238*pow(Omhh, 0.223)
    z_drag = 1291.*pow(Omhh, 0.251)/(1.+0.659*pow(Omhh, 0.828))*(1.+z_drag_b1*pow(Obhh,z_drag_b2))
    y_drag = z_equality/(1.+z_drag)

    # Set up free-streaming & infall growth function
    p_c = 0.25*(5.-np.sqrt(1.+24.*f_cdm))
    p_cb = 0.25*(5.-np.sqrt(1.+24.*f_cb))

    Omega_M_z = cosmo.Omega_M_z(z, CosPar)
    Omega_L_z = cosmo.Omega_L_z(z, CosPar)
    growth_k0 = z_equality*cosmo.D_growth(z, CosPar)
    growth_to_z0 = z_equality*cosmo.D_growth(0., CosPar)
    growth_to_z0 = growth_k0/growth_to_z0

    # Compute small-scale suppression
    alpha_nu = f_cdm/f_cb*(5.-2.*(p_c+p_cb))/(5.-4.*p_cb)*pow(1.+y_drag, p_cb-p_c)*(1.+f_bnu*(-0.553+0.126*f_bnu**2))/(1.-0.193*np.sqrt(f_hdm*n_degen_hdm)+0.169*f_hdm*pow(n_degen_hdm, 0.2))*(1.+(p_c-p_cb)/2.*(1.+1./(3.-4.*p_c)/(7.-4.*p_cb))/(1.+y_drag))
    alpha_gamma = np.sqrt(alpha_nu)
    beta_c = 1./(1.-0.949*f_bnu)

    ## Compute Transfer Functions
    #print "Computing transfer function..."

    qq = kk/Omhh*cosmo.Theta_CMB**2

    # Compute the scale-dependent growth functions
    y_freestream = 17.2*f_hdm*(1.+0.488*pow(f_hdm, -7./6.))*(n_degen_hdm*qq/f_hdm)**2
    temp1 = pow(growth_k0, 1.-p_cb)
    temp2 = pow(growth_k0/(1.+y_freestream), 0.7)
    growth_cb = pow(1.+temp2, p_cb/0.7)*temp1
    growth_cbnu = pow(pow(f_cb, 0.7/p_cb)+temp2, p_cb/0.7)*temp1

    # Compute the master function
    gamma_eff = Omhh*(alpha_gamma+(1.-alpha_gamma)/(1.+(0.43*kk*sound_horizon_fit)**4))
    qq_eff = qq*Omhh/gamma_eff

    tf_sup_L = np.log(np.e+1.84*beta_c*alpha_gamma*qq_eff)
    tf_sup_C = 14.4+325./(1.+60.5*pow(qq_eff, 1.11))
    tf_sup = tf_sup_L/(tf_sup_L+tf_sup_C*qq_eff**2)

    qq_nu = 3.92*qq*np.sqrt(n_degen_hdm/f_hdm)
    max_fs_correction = 1.+1.2*pow(f_hdm, 0.64)*pow(n_degen_hdm, 0.3+0.6*f_hdm)/(pow(qq_nu, -1.6)+pow(qq_nu, 0.8))
    tf_master = tf_sup*max_fs_correction

    # Compute the CDM+HDM+baryon transfer functions
    tf_cb = tf_master*growth_cb/growth_k0
    tf_cbnu = tf_master*growth_cbnu/growth_k0

    #print "Done computing transfer function."
#   return tf_sup
    return (tf_cb*growth_to_z0)**2*(kk**ns) #*(3E5/h/100.)**(ns+3.)

# Use dlogk
# R has to be scalar
# k and R are both in physical unit
def sigma_R_integrand_unnorm(logk, R, z, CosPar):
    kk = np.exp(logk)
    Tk = kk**3*EH_TF_mpc(kk, z, CosPar) #/2./np.pi**2*(3E5/70.)**(3.95)*(1.95E-5*pow(0.27, -0.785-0.05*np.log(0.27))*np.exp(-1.14))**2
    kR = kk*R.reshape(R.size, 1) #Be careful with the broadcasting rules 
    # Top-hat window function
    WRK = 3./kR**3*(np.sin(kR)-kR*np.cos(kR))
    return Tk*WRK**2

def sigma_R_sqr_unnorm(R, z, CosPar):
    RR = np.array(R)
    k_min = 1E-6
    k_max = 1E3
    dlogk = 1.E-2
    logk = np.arange(np.log(k_min),np.log(k_max)+2.*dlogk,dlogk)
    nlogk = logk.size
    Integrand = sigma_R_integrand_unnorm(logk, RR, z, CosPar)

    # Compute the integral (Maybe a vectorized trapezoidal integral would be faster?)
    #print "Integrating..."
    # return np.sum((Integrand[:,1:]+Integrand[:,:nlogk-1])/2.*dlogk, axis=1)
    return np.sum((Integrand[:,2:]+Integrand[:,:nlogk-2]+4.*Integrand[:,1:nlogk-1])/6.*dlogk, axis=1)

def sigma_j_R_integrand_unnorm(j, logk, R, z, CosPar):
    kk = np.exp(logk)
    Tk = kk**(3+2.*j)*EH_TF_mpc(kk, z, CosPar) #/2./np.pi**2*(3E5/70.)**(3.95)*(1.95E-5*pow(0.27, -0.785-0.05*np.log(0.27))*np.exp(-1.14))**2
    kR = kk*R.reshape(R.size, 1) #Be careful with the broadcasting rules 
    # Top-hat window function
    WRK = 3./kR**3*(np.sin(kR)-kR*np.cos(kR))
    return Tk*WRK**2

def sigma_j_R_sqr_unnorm(j, R, z, CosPar):
    RR = np.array(R)
    k_min = 1E-6
    k_max = 1E3
    dlogk = 1.E-2
    logk = np.arange(np.log(k_min),np.log(k_max)+2.*dlogk,dlogk)
    nlogk = logk.size
    Integrand = sigma_j_R_integrand_unnorm(j, logk, RR, z, CosPar)

    # Compute the integral (Maybe a vectorized trapezoidal integral would be faster?)
    #print "Integrating..."
    # return np.sum((Integrand[:,1:]+Integrand[:,:nlogk-1])/2.*dlogk, axis=1)
    return np.sum((Integrand[:,2:]+Integrand[:,:nlogk-2]+4.*Integrand[:,1:nlogk-1])/6.*dlogk, axis=1)

def sigma_j_R_sqr(j, R, z, CosPar):
    sjR_sqr_unnormalized = sigma_j_R_sqr_unnorm(j, R, z, CosPar)
    s8_sqr = sigma_R_sqr_unnorm(8./CosPar['h'], 0., CosPar)
    return sjR_sqr_unnormalized*CosPar['sigma_8']**2/s8_sqr

# Everything in physical unit, 
# kk in Mpc^-1
# Pk (output) in Mpc^3
# if you want kk in h Mpc^-1; Pk in h^-3 Mpc^3: do kk/h and P(k)*h^3 (Obvious, isn't it?)
def ps_linear(kk, z, CosPar):
    
    # Compute transfer function
    #print "Computing transfer function..."
    Tk = EH_TF_mpc(kk, z, CosPar) # (Tk*Dz/D0)^2*k^ns

    # Compute normalization
    #print "Computing normalziation..."
    # s8_sqr = sigma_R_sqr(8./CosPar['h'], z, CosPar) # Get normalization
    s8_sqr = sigma_R_sqr_unnorm(8./CosPar['h'], 0., CosPar) # Get normalization
    # print "COBE normalization:"
    # print np.sqrt((3E5/70.)**3.95*(1.94E-5*pow(0.27, -0.785-0.05*np.log(0.27))*np.exp(0.95*0.05-0.169*0.05**2))**2*s)
    return CosPar['sigma_8']**2*Tk/s8_sqr*2.*np.pi**2

def sigma_R_sqr(R, z, CosPar):
    sR_sqr_unnormalized = sigma_R_sqr_unnorm(R, z, CosPar)
    s8_sqr = sigma_R_sqr_unnorm(8./CosPar['h'], 0., CosPar)
    return sR_sqr_unnormalized*CosPar['sigma_8']**2/s8_sqr

# For PD96 and Smith03, k in h Mpc^-1 
def sigma_R_integrand_be84(logk, R, CosPar):
    kk = np.exp(logk)
    Tk = Deltak_linear_be84(kk, CosPar)#/2./np.pi**2*(3E5/70.)**(3.95)*(1.95E-5*pow(0.27, -0.785-0.05*np.log(0.27))*np.exp(-1.14))**2
    kR = kk*R.reshape(R.size, 1)
    # Top-hat window function
    WRK = 3./kR**3*(np.sin(kR)-kR*np.cos(kR))
    return Tk*WRK**2

def sigma_R_sqr_be84(R, CosPar):
    RR = np.array(R)
    k_min = 1E-6
    k_max = 1E3
    dlogk = 1.E-2
    logk = np.arange(np.log(k_min),np.log(k_max)+2.*dlogk,dlogk)
    nlogk = logk.size
    Integrand = sigma_R_integrand_be84(logk, RR, CosPar)

    # Compute the integral (Maybe a vectorized trapezoidal integral would be faster?)
    #print "Integrating..."
    #return np.sum((Integrand[:,1:]+Integrand[:,:nlogk-1])/2.*dlogk, axis=1)
    return np.sum((Integrand[:,2:]+Integrand[:,:nlogk-2]+4.*Integrand[:,1:nlogk-1])/6.*dlogk, axis=1)

# Linear power spectrum by Bond & Efstathiou 1984
# Output Delta(k)^2: unitless
def Deltak_linear_be84(kk, CosPar):
    print "IF YOU SEE ME TOO MANY TIMES, YOU SHOULD VECTORIZE YOUR CODE. MAYBE YOU SHOULD VECTORIZE ME!"
    gams = CosPar['Omega_M']*CosPar['h']
    sig8 = CosPar['sigma_8']
    p_index = CosPar['ns']
    keff = 0.172+0.011*np.log(gams/0.36)**2
    q = cosmo.EPS+kk/gams
    q8 = cosmo.EPS+keff/gams
    tk = 1./pow(1.+pow(6.4*q+pow(3.*q, 1.5)+pow(1.7*q, 2), 1.13), 1./1.13)
    tk8 = 1./pow(1.+pow(6.4*q8+pow(3.*q8, 1.5)+pow(1.7*q8, 2), 1.13), 1./1.13)
    return sig8**2*pow(q/q8, 3.+p_index)*tk**2/tk8**2

ps_linear_be84 = lambda kk, z, CosPar: (cosmo.D_growth(z, CosPar)/cosmo.D_growth(0,CosPar))**2*Deltak_linear_be84(kk, CosPar)*2.*np.pi**2/kk**3

# Non linear PS formula by Peacock & Dodds 1996
# Output
def ps_nl_pd96(kk, z, CosPar):
    k_min = 1E-5
    k_max = 1E2
    if min(kk)<k_min: raise ValueError("k too small")
    if max(kk)>k_max: raise ValueError("k too large")

    z_equality = cosmo.z_equality(CosPar)
    Omega_M_z = cosmo.Omega_M_z(z, CosPar)
    Omega_L_z = cosmo.Omega_L_z(z, CosPar)
    g = cosmo.D_growth(z, CosPar)
    g0 = cosmo.D_growth(0., CosPar)
    amp = g/g0

    # Calculate y and rn, y is linear spectrum, rn is the effective spectral index: rn(k)=dlnP(k/2)/dln(k/2)
    dlogk = 1.E-2
    logk = np.arange(np.log(k_min),np.log(k_max)+2.*dlogk,dlogk)
    nlogk = logk.size
    klin = np.exp(logk)

    # linear spectrum by BE84
    #y = amp**2*Deltak_linear_be84(klin, CosPar)
    y = ps_linear_be84(klin, z, CosPar)*klin**3

    # Effective spectral index
    y2 = Deltak_linear_be84(klin/2., CosPar)
    y2plus = Deltak_linear_be84(klin/2.*1.01, CosPar)
    rn = -3.+np.log(y2plus/y2)/np.log(1.01)

    # Fitting Formula
    a = 0.482*pow(1.+rn/3., -0.947)
    b = 0.226*pow(1.+rn/3., -1.778)
    alp = 3.310*pow(1.+rn/3., -0.244)
    bet = 0.862*pow(1.+rn/3., -0.287)
    vir = 11.55*pow(1.+rn/3., -0.423)
    Deltak_nl = y*pow((1.+b*y*bet+pow(a*y, alp*bet))/(1.+pow(pow(a*y, alp)*g**3/vir/np.sqrt(y), bet)), 1./bet)
    k_nl = klin*pow(1.+Deltak_nl, 1./3.)
    f = interpolate.interp1d(k_nl, Deltak_nl)
    return f(kk)/kk**3

# halofit by Smith et al. 2003
# correction (P-P_linear) = (P-P_linear)*(1.+2y**2)/(1.+y**2); y = k/10.
