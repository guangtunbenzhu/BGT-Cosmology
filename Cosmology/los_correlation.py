"""
los_correlation
"""
import numpy as np
from scipy import interpolate
import halomodel as hm
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

CosPar={'Omega_M':0.3, 'Omega_L':0.7, 'Omega_b':0.045, 'Omega_nu':1e-5, 'n_degen_nu':3., 'h':0.7, 'sigma_8':0.8, 'ns':0.96}
z = 0.52
Mhalo = 3E13

# def fwhm_los_xiR(y, Mhalo, z, CosPar)
#   """
#   fwhm_los_xiR(R, Mhalo, z, CosPar)
#   """

y_min = 1E0
y_max = 2E2
dy = 0.1
y = np.arange(y_min, y_max, dy)
ny = y.size

R_min = 1E-4
R_max = 4E2
dlogR = 1E-2
R = np.exp(np.arange(np.log(R_min)-2.*dlogR, np.log(R_max)+2.*dlogR, dlogR))
nR = R.size

xi_mon = hm.xi_Legendre_monopole(R, Mhalo, z, CosPar)
xi_qua, xi_hex = hm.xi_Legendre_multipole(R, Mhalo, z, CosPar)
int_mon = interpolate.interp1d(R, xi_mon.reshape(nR))
int_qua = interpolate.interp1d(R, xi_qua.reshape(nR))
int_hex = interpolate.interp1d(R, xi_hex.reshape(nR))

rp = np.ones(ny).reshape(1,ny)*y.reshape(ny,1)
ppi = y.reshape(1,ny)*np.ones(ny).reshape(ny,1)
rr = np.sqrt(rp**2+ppi**2)
mmu = ppi/rr
pp_mon = np.ones(ny*ny).reshape(ny,ny)
pp_qua = (3.*mmu*mmu-1.)/2.
pp_hex = (35.*mmu**4-30.*mmu**2+3.)/8.
xis = int_mon(rr)*pp_mon + int_qua(rr)*pp_qua + int_hex(rr)*pp_hex

# comoving
fwhm = np.zeros(ny)

for i in np.arange(ny/2):
    f_interpol = interpolate.interp1d(xis[i,::-1].reshape(ny), y[::-1])
    xi_max = max(xis[i,:])
    fwhm[i] = f_interpol(xi_max/2.)*2.

zz = np.arange(z, z+0.2, 0.001)
comoving_zz = hm.cosmo.comoving_distance(zz, CosPar)
dcomoving_zz = comoving_zz - comoving_zz[0]
dzz = zz - zz[0]

f_interpol = interpolate.interp1d(dcomoving_zz, dzz)
vdisp = f_interpol(fwhm/2.35)*hm.cosmo.speed_of_light

#plt.plot(y/(1.+z), vdisp)
