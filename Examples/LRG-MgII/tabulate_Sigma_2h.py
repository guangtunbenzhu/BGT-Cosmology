
import time
import sys
import numpy as np
import cosmology as cosmo
import halomodel as hm
from scipy import interpolate
from scipy import integrate

CosPar = {'Omega_M':0.3, 'Omega_L':0.7, 'Omega_b':0.045, 'Omega_nu':1e-5, 'n_degen_nu':3., 'h':0.7, 'sigma_8':0.8, 'ns':0.96}
z = 0.52
#data = np.genfromtxt('xiR_2h_no_bias_z0.52.dat', dtype=[('R', 'f'), ('xi', 'f')])
data = np.genfromtxt('linear_xiR_2h_no_bias_z0.52.dat', dtype=[('R', 'f'), ('xi', 'f')])
xi = data['xi']

R_min = 1E-5 
R_max = 3E2 # Can't be bigger than this because k_min = 1E-6
dlogR = 1.E-2
RR = np.exp(np.arange(np.log(R_min)-2.*dlogR,np.log(R_max)+2.*dlogR,dlogR))
nlogR = RR.size

# Sigma(y) = rho*Int{1.+[xi()]}
y_min = 2E-5
y_max = 1E2
dlogy = 1E-2
yy = np.exp(np.arange(np.log(y_min),np.log(y_max)+2.*dlogy,dlogy))
nlogy = yy.size

# RR is comoving
f = interpolate.interp1d(RR/(1.+z), xi)
# f = interpolate.interp1d(RR, xi)

Sigma_project_integrand = lambda r, y: 2.*f(r)*r**2/np.sqrt(r*r-y*y)

Sigma_y = np.zeros(nlogy)

# setup progress bar
progressbar_width = 80
progressbar_interval = nlogy/progressbar_width+1
sys.stdout.write("[%s]" % (" " * progressbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (progressbar_width+1)) # return to start of line, after '['

s_min_exclusion = hm.virial_radius(1E13, z, CosPar)
for i in np.arange(nlogy):
    if yy[i]>s_min_exclusion:
        s_min = yy[i]
    else:
        s_min = s_min_exclusion
#   s_max = s_min*6
#   if s_max > R_max/(1.+z):
#       s_max = R_max/(1.+z)
    s_max = R_max/(1.+z)
    ss = np.exp(np.arange(np.log(s_min)+dlogy,np.log(s_max)+dlogy,dlogy))
    nlogs = ss.size
    Integrand = Sigma_project_integrand(ss, s_min)
    Sigma_y[i] = np.sum((Integrand[2:]+Integrand[:nlogs-2]+4.*Integrand[1:nlogs-1])/6.*dlogy)
    if (i%progressbar_interval==0):
       sys.stdout.write("-")
       sys.stdout.flush()
sys.stdout.write("\n")

Sigma_y = cosmo.rho_critical(z, CosPar)*cosmo.Omega_M_z(z, CosPar)*Sigma_y/1E12
# Sigma_y = cosmo.rho_critical(z, CosPar)*cosmo.Omega_M_z(z, CosPar)*Sigma_y/1E12/(1.+z)**3

#Sigma_func = np.vectorize(lambda y:
#       integrate.quad(Sigma_project_integrand, y, R_max, limit=1000,
#       args=(y)))
#Sigma_y = cosmo.rho_critical(z, CosPar)*cosmo.Omega_M_z(z, CosPar)*Sigma_func(yy)

#np.savetxt('SigmaR_2h_no_bias_z0.52.dat', zip(yy, Sigma_y), fmt='%G  %G')
np.savetxt('linear_SigmaR_2h_no_bias_z0.52.dat', zip(yy, Sigma_y), fmt='%G  %G')
