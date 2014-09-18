
""" 
Useful small tools for spectroscopic analysis
"""

import numpy as np
import logging
from scipy import integrate

# create a center wavelength grid with constant width in log (i.e., velocity) space:
get_loglambda = lambda minwave=900., maxwave=14400., dloglam=1.E-4: pow(10,np.arange(np.log10(minwave), np.log10(maxwave)+dloglam, dloglam))

def get_loglambda(minwave=900., maxwave=14400., dloglam=1.E-4):
    """
    create a center wavelength grid with constant width in log (i.e., velocity) space:
        get_loglambda(minwave=minwave, maxwave=maxwave, dloglam=dloglam)
    """
    logging.info("default minwave=900 A, maxwave=14400 A, dloglam=1E-4")
    return np.arange(np.log10(minwave), np.log10(maxwave)+dloglam, dloglam)


# Useful Constants
EPS = 1.E-10  # Some small number
speed_of_light = 299792.458 # km/s
MSun = 1.9891E30 # kg
GG = 6.67384E-11 # m^3 kg^-1 s^-2 or N (m/kg)^2
GG_MSun = 4.302E-9 # Mpc km^2 s^-2 MSun^-1
Theta_CMB = 2.728/2.7 # T(CMB)/2.7
parsec = 3.0857E16 # m
MProton = 1.6726231E-27 # kg
MNeutron =  1.674920E-27 # kg

# Basic Functions
Hubble_z = lambda z, CosPar: CosPar['h']*np.sqrt(CosPar['Omega_L']+(1.+z)**2*(1.-CosPar['Omega_M']-CosPar['Omega_L']+CosPar['Omega_M']*(1.+z)))
Omega_M_z = lambda z, CosPar: CosPar['Omega_M']*(1.+z)**3*CosPar['h']**2/Hubble_z(z, CosPar)**2
Omega_L_z = lambda z, CosPar: CosPar['Omega_L']*CosPar['h']**2/Hubble_z(z, CosPar)**2
z_equality = lambda CosPar: 2.5E4*CosPar['Omega_M']*CosPar['h']**2/Theta_CMB**4 #Actually 1+z_eq
k_equality = lambda CosPar: 0.0746*CosPar['Omega_M']*CosPar['h']**2/Theta_CMB**2
k_Silk = lambda CosPar: 1.6**pow(CosPar['Omega_b']*CosPar['h']**2, 0.52)*pow(CosPar['Omega_M']*CosPar['h']**2, 0.73)*(1.+pow(10.4*CosPar['Omega_M']*CosPar['h']**2, -0.95))
sound_horizon = lambda CosPar: 44.5*np.log(9.83/(CosPar['Omega_M']*CosPar['h']**2))/np.sqrt(1.+10.*pow(CosPar['Omega_b']*CosPar['h']**2, 0.75)) # Mpc
rho_critical_SI = lambda z, CosPar: 3E4/8./np.pi/GG*Hubble_z(z,CosPar)**2
rho_critical = lambda z, CosPar: 3E4/8./np.pi/GG_MSun*Hubble_z(z, CosPar)**2

def D_growth(z, CosPar):
    Omega_M = Omega_M_z(z,CosPar)
    Omega_L = Omega_L_z(z,CosPar)
    return 2.5*Omega_M/(pow(Omega_M, 4./7.)-Omega_L+(1.+Omega_M/2.)*(1.+Omega_L/70.))/(1.+z)

# Distance Modules
# OmegaM and OmegaL have to be scalars
ez_integrand = lambda z, Omega_M, Omega_L: 1./np.sqrt((1.+z)**2*(1.+Omega_M*z)-z*(2.+z)*Omega_L)
def comoving_distance(z, CosPar):
    comdist_func = np.vectorize(lambda z, Omega_M, Omega_L:
        integrate.quad(ez_integrand, 0., z, limit=1000, 
        args=(Omega_M, Omega_L)))
    comdist, err = comdist_func(z, CosPar['Omega_M'], CosPar['Omega_L'])

    Omega_k = 1.-CosPar['Omega_M']-CosPar['Omega_L']
    if Omega_k > EPS:
        return np.sinh(np.sqrt(Omega_k)*comdist)/np.sqrt(Omega_k)*speed_of_light/CosPar['h']/100.
    elif Omega_k < (-EPS):
        return np.sin(np.sqrt(-Omega_k)*comdist)/np.sqrt(-Omega_k)*speed_of_light/CosPar['h']/100.
    else:
        return comdist*speed_of_light/CosPar['h']/100.
 
luminosity_distance = lambda z, CosPar: comoving_distance(z, CosPar)*(1.+z)
angdiameter_distance = lambda z, CosPar: comoving_distance(z, CosPar)/(1.+z)

