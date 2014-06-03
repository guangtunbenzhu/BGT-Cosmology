"""
Sigma_MonteCarlo
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
lrg = np.genfromtxt('LRG-MgII.txt', dtype=[('R','f'), ('npairs', 'f'), ('W','f'), ('Werr', 'f')])
vdisp = np.genfromtxt('LRG-MgII_vdisp.txt', dtype=[('R','f'), ('npairs', 'f'), ('vdisp','f'), ('vdisp_err', 'f')])
Sigma_2h0 = np.genfromtxt('linear_SigmaR_2h_no_bias_z0.52.dat', dtype=[('R', 'f'), ('Sigma_R', 'f')])

# turn REW to surface density (minimum) in MSun/pc^2
pccm = 3.08567758E18 #parsec to cm
factor = 1.13E20/0.3030/(2803.53**2)
Mmg24 = 24.305*1.67E-27
Msolar = 2.E30
mass_factor = factor*Mmg24/Msolar*pccm*pccm/2/1E3*1E8 #3 is for Mg II 2803; 1E3 is for mA; 1E9 is for display purpose

#fbaryon = 0.167*0.0018

Mhalo_min = 1.E11
Mhalo_max1 = 5.E12
dlogM = 2E-1
Mhalo1 = np.exp(np.arange(np.log(Mhalo_min), np.log(Mhalo_max1), dlogM))
Mhalo_min2 = 1.E14
Mhalo_max2 = 3.E15
dlogM = 2E-1
Mhalo2 = np.exp(np.arange(np.log(Mhalo_min2), np.log(Mhalo_max2), dlogM))
Mhalo_min3 = 5.E12
Mhalo_max = 1.E14
dlogM = 2E-2
Mhalo3 = np.exp(np.arange(np.log(Mhalo_min3), np.log(Mhalo_max), dlogM))
Mhalo = np.concatenate([Mhalo1, Mhalo3, Mhalo2])

#Mhalo = np.exp(np.arange(np.log(Mhalo_min), np.log(Mhalo_max), dlogM))
nMhalo = Mhalo.shape[0]
y = lrg['R']
Sigma_lrg = lrg['W']*mass_factor
Sigma_lrg_err = lrg['Werr']*mass_factor
vdisp_lrg = vdisp['vdisp']
vdisp_lrg_err = vdisp['vdisp_err']
ny = y.shape[0]
for i in np.arange(4): 
    Sigma_lrg[i] = Sigma_lrg[i]*2/2.
    Sigma_lrg_err[i] = Sigma_lrg_err[i]*2./2.
#for i in np.arange(2)-3: 
#   Sigma_lrg[i] = Sigma_lrg[i]*1.5
#   Sigma_lrg_err[i] = Sigma_lrg_err[i]*1.5
#Sigma_lrg_err[14:15] = Sigma_lrg_err[14:15]*30.
#Sigma_lrg_err[13] = Sigma_lrg_err[13]*10.
#Sigma_lrg_err[14] = Sigma_lrg_err[14]*10.
#Sigma_lrg_err[15] = Sigma_lrg_err[15]*10.

# 1-halo Sigma_1h(ny, nMhalo)
Sigma_1h = hm.NFW_project_profile(y, Mhalo, z, CosPar)

# 2-halo Sigma_1h(ny, nMhalo)
bM = hm.bias(Mhalo, z, CosPar)
f = interpolate.interp1d(Sigma_2h0['R'], Sigma_2h0['Sigma_R'])
Sigma_2h = bM*f(y).reshape(ny,1)

A_min = 2E-2
A_max = 2E+2
dlogA = 1E-1
A = np.exp(np.arange(np.log(A_min), np.log(A_max), dlogA))
nA = A.shape[0]

# Sigma_all[ny, nA_1h, nA_2h, nMhalo)
Sigma_all = np.ones((1, 1, nA, 1))*(A.reshape(1, nA, 1, 1)*Sigma_1h.reshape(ny, 1, 1, nMhalo)) + np.ones((1, nA, 1, 1))*(A.reshape(1, 1, nA, 1)*Sigma_2h.reshape(ny, 1, 1, nMhalo))

# Calculate Chi-square Chi2(nA_1h, nA_2h, nMhalo)
Chi2 = np.sum((Sigma_all-Sigma_lrg.reshape(ny, 1, 1, 1))**2/Sigma_lrg_err.reshape(ny, 1, 1, 1)**2, axis=0)
Likelihood = np.exp(-Chi2/2.)
Likelihood = Likelihood/Likelihood.sum()

joint_1h_Mhalo = np.sum(Likelihood, axis=1) # (nA_1h, nMhalo)
joint_2h_Mhalo = np.sum(Likelihood, axis=0) # (nA_2h, nMhalo)
joint_1h_2h = np.sum(Likelihood, axis=2) # (nA_1h, nA_2h)

# Numerical Recipes Page 697, Chapter 15.6
levels_3d = Likelihood.max()*np.array([np.exp(-3.53/2.), np.exp(-8.02/2.), np.exp(-14.2/2.)])
levels_1h_Mhalo = joint_1h_Mhalo.max()*np.array([np.exp(-2.30/2.), np.exp(-6.17/2.), np.exp(-11.8/2.)])
levels_2h_Mhalo = joint_2h_Mhalo.max()*np.array([np.exp(-2.30/2.), np.exp(-6.17/2.), np.exp(-11.8/2.)])
levels_1h_Mhalo_f = joint_1h_Mhalo.max()*np.array([1., np.exp(-2.30/2.), np.exp(-6.17/2.), np.exp(-11.8/2.)])
levels_2h_Mhalo_f = joint_2h_Mhalo.max()*np.array([1., np.exp(-2.30/2.), np.exp(-6.17/2.), np.exp(-11.8/2.)])
levels_1h_2h = joint_1h_2h.max()*np.array([np.exp(-2.30/2.), np.exp(-6.17/2.), np.exp(-11.8/2.)])

# joint_1h_Mhalo
xmax_1h = np.argmax(joint_1h_Mhalo, axis=1)
ymax_1h = np.zeros(nA)
for i in np.arange(nA): ymax_1h[i] = joint_1h_Mhalo[i,xmax_1h[i]]
iAmax_1h = np.argmax(ymax_1h)
iMmax_1h = xmax_1h[iAmax_1h]
Max_likehood_1h = joint_1h_Mhalo[iAmax_1h, iMmax_1h]

# joint_2h_Mhalo
xmax_2h = np.argmax(joint_2h_Mhalo, axis=1)
ymax_2h = np.zeros(nA)
for i in np.arange(nA): ymax_2h[i] = joint_2h_Mhalo[i,xmax_2h[i]]
iAmax_2h = np.argmax(ymax_2h)
iMmax_2h = xmax_2h[iAmax_2h]
Max_likehood_2h = joint_2h_Mhalo[iAmax_2h, iMmax_2h]

print Mhalo[iMmax_1h], Mhalo[iMmax_2h]
print np.argmin(Chi2)/(nA*nMhalo), np.argmin(Chi2)%(nA*nMhalo)/nMhalo, np.argmin(Chi2)%(nA*nMhalo)%nMhalo

# Mpc 
R_min = 2E-5
R_max = 9E1
dlogR = 1E-2
RR = np.exp(np.arange(np.log(R_min), np.log(R_max)+dlogR, dlogR))

iAmax_1h_all = np.argmin(Chi2)/(nA*nMhalo)
iAmax_2h_all = np.argmin(Chi2)%(nA*nMhalo)/nMhalo
iMmax_all = np.argmin(Chi2)%(nA*nMhalo)%nMhalo
Amax_1h = A[iAmax_1h_all]
Amax_2h = A[iAmax_2h_all]
Mmax = Mhalo[iMmax_all]

# Get 1sigma errors
minChi2_Mhalo = np.zeros(nMhalo)
minChi2_1h = np.zeros(nA)
minChi2_2h = np.zeros(nA)
for i in np.arange(nMhalo):
    minChi2_Mhalo[i] = np.min(Chi2[:,:,i])
for i in np.arange(nA):
    minChi2_1h[i] = np.min(Chi2[i,:,:])
    minChi2_2h[i] = np.min(Chi2[:,i,:])

#levels_3d = Likelihood.max()*np.array([np.exp(-3.53/2.), np.exp(-8.02/2.), np.exp(-14.2/2.)])
iMmax_all_tmp = np.argmin(minChi2_Mhalo)
iAmax_1h_all_tmp = np.argmin(minChi2_1h)
iAmax_2h_all_tmp = np.argmin(minChi2_2h)
print iMmax_all_tmp, iMmax_all
print iAmax_1h_all_tmp, iAmax_1h_all
print iAmax_2h_all_tmp, iAmax_2h_all
print minChi2_Mhalo[iMmax_all_tmp], minChi2_1h[iAmax_1h_all_tmp], minChi2_2h[iAmax_2h_all_tmp]
minChi2_all = minChi2_Mhalo[iMmax_all_tmp]

iMmax_all_left = np.argmin(np.abs(minChi2_Mhalo[:iMmax_all_tmp]-minChi2_all-1.))
iMmax_all_right = np.argmin(np.abs(minChi2_Mhalo[iMmax_all_tmp:]-minChi2_all-1.))+iMmax_all_tmp
iAmax_1h_all_left = np.argmin(np.abs(minChi2_1h[:iAmax_1h_all_tmp]-minChi2_all-1.))
iAmax_1h_all_right = np.argmin(np.abs(minChi2_1h[iAmax_1h_all_tmp:]-minChi2_all-1.))+iAmax_1h_all_tmp
iAmax_2h_all_left = np.argmin(np.abs(minChi2_2h[:iAmax_2h_all_tmp]-minChi2_all-1.))
iAmax_2h_all_right = np.argmin(np.abs(minChi2_2h[iAmax_2h_all_tmp:]-minChi2_all-1.))+iAmax_2h_all_tmp
print np.log10(Mhalo[iMmax_all_left]), np.log10(Mhalo[iMmax_all_tmp]), np.log10(Mhalo[iMmax_all_right])
print np.log10(A[iAmax_1h_all_left]), np.log10(A[iAmax_1h_all_tmp]), np.log10(A[iAmax_1h_all_right])
print np.log10(A[iAmax_2h_all_left]), np.log10(A[iAmax_2h_all_tmp]), np.log10(A[iAmax_2h_all_right])
print np.log10(Mhalo[iMmax_all_tmp])-np.log10(Mhalo[iMmax_all_left]), np.log10(Mhalo[iMmax_all_right])-np.log10(Mhalo[iMmax_all_tmp])
print np.log10(A[iAmax_1h_all_tmp])-np.log10(A[iAmax_1h_all_left]), np.log10(A[iAmax_1h_all_right])-np.log10(A[iAmax_1h_all_tmp])
print np.log10(A[iAmax_2h_all_tmp])-np.log10(A[iAmax_2h_all_left]), np.log10(A[iAmax_2h_all_right])-np.log10(A[iAmax_2h_all_tmp])

iAmax_1h_12 = np.argmin(Chi2[:,:,0])/(nA)
iAmax_2h_12 = np.argmin(Chi2[:,:,0])%(nA)
Amax_1h_12 = A[iAmax_1h_12]
Amax_2h_12 = A[iAmax_2h_12]
M12 = Mhalo[0]
print iAmax_1h_12, iAmax_2h_12

iAmax_1h_15 = np.argmin(Chi2[:,:,nMhalo-1])/(nA)
iAmax_2h_15 = np.argmin(Chi2[:,:,nMhalo-1])%(nA)
Amax_1h_15 = A[iAmax_1h_15]
Amax_2h_15 = A[iAmax_2h_15]
M15 = Mhalo[nMhalo-1]
print iAmax_1h_15, iAmax_2h_15

bM_12 = bM[0]
Sigma_1h_12 = hm.NFW_project_profile(RR, M12, z, CosPar)
Sigma_2h_12 = bM_12*f(RR).reshape(RR.size,1)
# This is Sigma(R), we also need Sigma(<R)
Sigma_all_12 = (Amax_1h_12*Sigma_1h_12+Amax_2h_12*Sigma_2h_12).reshape(RR.size)

Mtmp=np.log10(2.26E13)
iMtmp = np.argmin(np.abs(Mtmp-np.log10(Mhalo)))
bM_sm6 = bM[iMtmp]
Chi2_joint_1h_2h_sm6 = Chi2[:,:,iMtmp] # (nA_1h, nA_2h)
# joint_1h_2h_sm6
xmax_1h_2h_sm6 = np.argmin(Chi2_joint_1h_2h_sm6, axis=1)
ymax_1h_2h_sm6 = np.zeros(nA)
for i in np.arange(nA): ymax_1h_2h_sm6[i] = Chi2_joint_1h_2h_sm6[i,xmax_1h_2h_sm6[i]]
iAmax_1h_sm6 = np.argmin(ymax_1h_2h_sm6)
iAmax_2h_sm6 = xmax_1h_2h_sm6[iAmax_1h_sm6]
minChi2_1h_sm6 = np.zeros(nA)
minChi2_2h_sm6 = np.zeros(nA)
for i in np.arange(nA):
    minChi2_1h_sm6[i] = np.min(Chi2_joint_1h_2h_sm6[i,:])
    minChi2_2h_sm6[i] = np.min(Chi2_joint_1h_2h_sm6[:,i])
iAmax_1h_sm6_left = np.argmin(np.abs(minChi2_1h_sm6[:iAmax_1h_sm6]-Chi2_joint_1h_2h_sm6[iAmax_1h_sm6, iAmax_2h_sm6]-6.63))
iAmax_2h_sm6_left = np.argmin(np.abs(minChi2_2h_sm6[:iAmax_2h_sm6]-Chi2_joint_1h_2h_sm6[iAmax_1h_sm6, iAmax_2h_sm6]-6.63))
iAmax_1h_sm6_right = np.argmin(np.abs(minChi2_1h_sm6[iAmax_1h_sm6:]-Chi2_joint_1h_2h_sm6[iAmax_1h_sm6, iAmax_2h_sm6]-6.63))+iAmax_1h_sm6
iAmax_2h_sm6_right = np.argmin(np.abs(minChi2_2h_sm6[iAmax_2h_sm6:]-Chi2_joint_1h_2h_sm6[iAmax_1h_sm6, iAmax_2h_sm6]-6.63))+iAmax_2h_sm6
#iAmax_1h_sm6_left = np.argmin(np.abs(minChi2_1h_sm6[:iAmax_1h_sm6]-Chi2_joint_1h_2h_sm6[iAmax_1h_sm6, iAmax_2h_sm6]-1.))
#iAmax_2h_sm6_left = np.argmin(np.abs(minChi2_2h_sm6[:iAmax_2h_sm6]-Chi2_joint_1h_2h_sm6[iAmax_1h_sm6, iAmax_2h_sm6]-1.))
#iAmax_1h_sm6_right = np.argmin(np.abs(minChi2_1h_sm6[iAmax_1h_sm6:]-Chi2_joint_1h_2h_sm6[iAmax_1h_sm6, iAmax_2h_sm6]-1.))+iAmax_1h_sm6
#iAmax_2h_sm6_right = np.argmin(np.abs(minChi2_2h_sm6[iAmax_2h_sm6:]-Chi2_joint_1h_2h_sm6[iAmax_1h_sm6, iAmax_2h_sm6]-1.))+iAmax_2h_sm6

Sigma_1h_sm6 = (hm.NFW_project_profile(RR, 2.26E13, 0.16, CosPar)).reshape(RR.size)
Sigma_1h_sm6_highz = (hm.NFW_project_profile(RR, 2.26E13, z, CosPar)).reshape(RR.size)
Sigma_2h_sm6 = (bM_sm6*f(RR).reshape(RR.size,1)).reshape(RR.size)
# This is Sigma(R), we also need Sigma(<R)
Sigma_all_sm6 = (Sigma_1h_sm6+Sigma_2h_sm6).reshape(RR.size)
Sigma_all_sm6_highz = (Sigma_1h_sm6_highz+Sigma_2h_sm6).reshape(RR.size)

bM_15 = bM[nMhalo-1]
Sigma_1h_15 = hm.NFW_project_profile(RR, M15, z, CosPar)
Sigma_2h_15 = bM_15*f(RR).reshape(RR.size,1)
# This is Sigma(R), we also need Sigma(<R)
Sigma_all_15 = (Amax_1h_15*Sigma_1h_15+Amax_2h_15*Sigma_2h_15).reshape(RR.size)

bMmax = bM[iMmax_all]
Sigma_1h_max = hm.NFW_project_profile(RR, Mmax, z, CosPar)
Sigma_2h_max = bMmax*f(RR).reshape(RR.size,1)
# This is Sigma(R), we also need Sigma(<R)
Sigma_all_max = (Amax_1h*Sigma_1h_max+Amax_2h*Sigma_2h_max).reshape(RR.size)
Sigma_all_max_1h = (Amax_1h*Sigma_1h_max).reshape(RR.size)
Sigma_all_max_2h = (Amax_2h*Sigma_2h_max).reshape(RR.size)

# velocity dispersion
#vdisp_mu = 3.5
#vdisp_1h_max_dm = np.sqrt(Sigma_1h_max/(Sigma_1h_max+Sigma_2h_max)).reshape(RR.size)*(hm.NFW_project_sigma(RR, Mmax, z, CosPar)).reshape(RR.size)
#vdisp_2h_max = np.sqrt(Sigma_2h_max/(Sigma_1h_max+Sigma_2h_max)).reshape(RR.size)*(hm.NFW_project_2h_sigma(RR, Mmax, z, CosPar)).reshape(RR.size)
#vdisp_all_max = np.sqrt(vdisp_1h_max**2+vdisp_2h_max**2)
#vdisp_1h_max = np.sqrt(Sigma_all_max_1h/Sigma_all_max)*(hm.NFW_project_sigma(RR, Mmax, z, CosPar)).reshape(RR.size)/vdisp_mu
#vdisp_2h_max = np.sqrt(Sigma_all_max_2h/Sigma_all_max)*(hm.NFW_project_2h_sigma(RR, Mmax, z, CosPar)).reshape(RR.size)
#vdisp_all_max = np.sqrt(vdisp_1h_max**2+vdisp_2h_max**2)

# Calculate average Sigma (for sm6):
total_mass_sm6 = np.zeros(RR.size)
total_mass_1h_sm6 = np.zeros(RR.size)
total_mass_2h_sm6 = np.zeros(RR.size)
total_mass_sm6[0] = Sigma_all_sm6[0]*np.pi*R_min*R_min*1.0
total_mass_1h_sm6[0] = Sigma_1h_sm6[0]*np.pi*R_min*R_min*1.0
total_mass_2h_sm6[0] = Sigma_2h_sm6[0]*np.pi*R_min*R_min*1.0

# Trapezoidal integration requires small stepsize
# Simpson's integration is twice as accurate with the same stepsize
for i in np.arange(RR.size-1)+1:
        total_mass_sm6[i] = total_mass_sm6[i-1]+(Sigma_all_sm6[i]*RR[i]+Sigma_all_sm6[i-1]*RR[i-1])*(RR[i]-RR[i-1])*np.pi
        total_mass_1h_sm6[i] = total_mass_1h_sm6[i-1]+(Sigma_1h_sm6[i]*RR[i]+Sigma_1h_sm6[i-1]*RR[i-1])*(RR[i]-RR[i-1])*np.pi
        total_mass_2h_sm6[i] = total_mass_2h_sm6[i-1]+(Sigma_2h_sm6[i]*RR[i]+Sigma_2h_sm6[i-1]*RR[i-1])*(RR[i]-RR[i-1])*np.pi
# Needs testing
Sigma_average_density_sm6 = total_mass_sm6/np.pi/RR**2
Sigma_average_density_1h_sm6= total_mass_1h_sm6/np.pi/RR**2
Sigma_average_density_2h_sm6 = total_mass_2h_sm6/np.pi/RR**2
#intepolation
ff_sm6 = interpolate.interp1d(RR, Sigma_average_density_sm6)
ff_1h_sm6 = interpolate.interp1d(RR, Sigma_average_density_1h_sm6)
ff_2h_sm6 = interpolate.interp1d(RR, Sigma_average_density_2h_sm6)
Sigma_average_density_sm6_lrg  = ff_sm6(y)
Sigma_average_density_sm6_lrg_1h  = ff_1h_sm6(y)
Sigma_average_density_sm6_lrg_2h  = ff_2h_sm6(y)
gg_sm6 = interpolate.interp1d(RR, Sigma_all_sm6)
gg_1h_sm6 = interpolate.interp1d(RR, Sigma_1h_sm6)
gg_2h_sm6 = interpolate.interp1d(RR, Sigma_2h_sm6)
Sigma_all_sm6_lrg = gg_sm6(y)
Sigma_all_sm6_lrg_1h = gg_1h_sm6(y)
Sigma_all_sm6_lrg_2h = gg_2h_sm6(y)

DSigma_all_sm6 = (Sigma_average_density_sm6 - Sigma_all_sm6)
DSigma_all_sm6_1h = (Sigma_average_density_1h_sm6 - Sigma_1h_sm6)
DSigma_all_sm6_2h = (Sigma_average_density_2h_sm6 - Sigma_2h_sm6)
DSigma_all_sm6_lrg = (Sigma_average_density_sm6_lrg - Sigma_lrg)
DSigma_all_sm6_lrg_err = Sigma_lrg_err

# Calculate average Sigma:
total_mass = np.zeros(RR.size)
total_mass_1h = np.zeros(RR.size)
total_mass_2h = np.zeros(RR.size)
total_mass[0] = Sigma_all_max[0]*np.pi*R_min*R_min*1.0
total_mass_1h[0] = Sigma_all_max_1h[0]*np.pi*R_min*R_min*1.0
total_mass_2h[0] = Sigma_all_max_2h[0]*np.pi*R_min*R_min*1.0

# Trapezoidal integration requires small stepsize
# Simpson's integration is twice as accurate with the same stepsize
for i in np.arange(RR.size-1)+1:
        total_mass[i] = total_mass[i-1]+(Sigma_all_max[i]*RR[i]+Sigma_all_max[i-1]*RR[i-1])*(RR[i]-RR[i-1])*np.pi
        total_mass_1h[i] = total_mass_1h[i-1]+(Sigma_all_max_1h[i]*RR[i]+Sigma_all_max_1h[i-1]*RR[i-1])*(RR[i]-RR[i-1])*np.pi
        total_mass_2h[i] = total_mass_2h[i-1]+(Sigma_all_max_2h[i]*RR[i]+Sigma_all_max_2h[i-1]*RR[i-1])*(RR[i]-RR[i-1])*np.pi
# Needs testing
Sigma_average_density_max = total_mass/np.pi/RR**2
Sigma_average_density_max_1h = total_mass_1h/np.pi/RR**2
Sigma_average_density_max_2h = total_mass_2h/np.pi/RR**2
#intepolation
ff = interpolate.interp1d(RR, Sigma_average_density_max)
ff_1h = interpolate.interp1d(RR, Sigma_average_density_max_1h)
ff_2h = interpolate.interp1d(RR, Sigma_average_density_max_2h)
Sigma_average_density_max_lrg  = ff(y)
Sigma_average_density_max_lrg_1h  = ff_1h(y)
Sigma_average_density_max_lrg_2h  = ff_2h(y)
gg = interpolate.interp1d(RR, Sigma_all_max)
gg_1h = interpolate.interp1d(RR, Sigma_all_max_1h)
gg_2h = interpolate.interp1d(RR, Sigma_all_max_2h)
Sigma_all_max_lrg = gg(y)
Sigma_all_max_lrg_1h = gg_1h(y)
Sigma_all_max_lrg_2h = gg_2h(y)

DSigma_all_max = (Sigma_average_density_max - Sigma_all_max)/Amax_2h
DSigma_all_max_1h = (Sigma_average_density_max_1h - Sigma_all_max_1h)/Amax_2h
DSigma_all_max_2h = (Sigma_average_density_max_2h - Sigma_all_max_2h)/Amax_2h
DSigma_all_max_lrg = (Sigma_average_density_max_lrg - Sigma_lrg)/Amax_2h
DSigma_all_max_lrg_err = Sigma_lrg_err/Amax_2h

plt.clf()
#plt.figure(figsize=(10,6))
plt.subplots_adjust(left=0.20, bottom=0.2)
dashes1 = (18,7)
dashes2 = (8,3.1)
plt.loglog(RR, Amax_1h_12*Sigma_1h_12+Amax_2h_12*Sigma_2h_12, 'm--', dashes=dashes1, lw=4)
plt.loglog(RR, Amax_1h_15*Sigma_1h_15+Amax_2h_15*Sigma_2h_15, '--', dashes=dashes2, color='#FF8C00', lw=4)
legend_max = r'$\log_{10}\boldsymbol{\mathrm{M}}_\mathrm{halo}/\boldsymbol{\mathrm{M}}_\odot=14\ (\boldsymbol{\chi}^2/{dof}='+'%4.2f' % (Chi2[iAmax_1h_all, iAmax_2h_all, iMmax_all]/14.)+')$'
legend_12 = r'$\log_{10}\boldsymbol{\mathrm{M}}_\mathrm{halo}/\boldsymbol{\mathrm{M}}_\odot=12.0\ (\boldsymbol{\chi}^2/{dof}='+'%4.2f' % (Chi2[iAmax_1h_12, iAmax_2h_12, 0]/14.)+')$'
legend_15 = r'$\log_{10}\boldsymbol{\mathrm{M}}_\mathrm{halo}/\boldsymbol{\mathrm{M}}_\odot=15.5\ (\boldsymbol{\chi}^2/{dof}='+'%4.2f' % (Chi2[iAmax_1h_15, iAmax_2h_15, nMhalo-1]/14.)+')$'
plt.legend([legend_12, legend_15], handlelength=3.4, frameon=False, loc=3)
plt.loglog(RR, Amax_1h*Sigma_1h_max+Amax_2h*Sigma_2h_max, 'b', lw=3)
plt.errorbar(y, Sigma_lrg, yerr=Sigma_lrg_err, capsize=5, fmt='bo', mec='b', ms=10)
#plt.loglog(RR, Amax_1h_12*Sigma_1h_12+Amax_2h_12*Sigma_2h_12, 'm--', dashes=dashes1, lw=3)
#plt.loglog(RR, Amax_1h_15*Sigma_1h_15+Amax_2h_15*Sigma_2h_15, '--', dashes=dashes2, color='#FF8C00', lw=3)
#plt.errorbar(y[0:4], Sigma_lrg[0:4]/1.2, yerr=Sigma_lrg_err[0:4]/1.2, capsize=5, fmt='bo', mec='b', ms=10, mfc=None)
plt.xlabel(r'$\mathbf{r}_\mathrm{p}$', fontsize=24)
plt.ylabel(r'$\boldsymbol{\Sigma}_\mathrm{Mg\,II}\ (10^{-9} \mathrm{M}_\odot\ \mathrm{pc^{-2}})$', rotation='vertical', fontsize=26)
plt.xlim(1E-2, 5E1)
plt.ylim(5E-1, 4E3)
plt.loglog([0.015,0.025], [1E0, 1E0],'b', lw=3)
plt.text(1E0, 18E2, r'$\log_{10} \boldsymbol{\mathrm{M}}_\mathrm{halo}/\boldsymbol{\mathrm{M}}_\odot='+'%4.1f' % np.log10(Mhalo[iMmax_all])+'$', fontsize=16)
plt.text(1E0, 9E2, r'$\boldsymbol{\chi}^2/{dof}='+'%4.2f' % (Chi2[iAmax_1h_all, iAmax_2h_all, iMmax_all]/14.)+'$', fontsize=16)
plt.show()
plt.savefig('Halomodel_comparison_sat1.eps')

# Read in Mandelbaum
sm6 = np.genfromtxt('mandelbaum/2006/fig1/rebin.sm.all.sm6.highfdev.dat', dtype=[('R','f'), ('DSigma', 'f'), ('DSigma_err','f')])
sm7 = np.genfromtxt('mandelbaum/2006/fig1/rebin.sm.all.sm7.highfdev.dat', dtype=[('R','f'), ('DSigma', 'f'), ('DSigma_err','f')])
sm6_z = 0.16
sm7_z = 0.19
lf6 = np.genfromtxt('mandelbaum/2006/fig2/rebin.lum.all.L6faint.highfdev.dat', dtype=[('R','f'), ('DSigma', 'f'), ('DSigma_err','f')])
lf6_z = 0.2

plt.figure(figsize=(10,7))
plt.clf()
plt.subplots_adjust(left=0.20, bottom=0.2)
show_lrg_err = np.zeros(Sigma_lrg.size)
for i in np.arange(Sigma_lrg.size): show_lrg_err[i] = min(DSigma_all_max_lrg[i]*0.9999, DSigma_all_max_lrg_err[i])

plt.loglog(RR, DSigma_all_max, 'b', lw=3)
plt.loglog(RR, DSigma_all_max_1h, 'g', ls='--', dashes=dashes1, lw=3)
plt.loglog(RR, DSigma_all_max_2h, color='#FF8C00', ls='--', dashes=dashes2, lw=3) 
plt.legend([r"\textbf{Halo model}",r'\textbf{1-halo term}', r'\textbf{2-halo term}'], handlelength=3.5, frameon=False, loc=3)#, fontsize=14)

plt.errorbar(y, DSigma_all_max_lrg, yerr=show_lrg_err, capsize=5, fmt='o', color='b', mec='b', ms=10)
#plt.errorbar(sm6['R']/1E3/0.7/(1.+0.1), sm6['DSigma']*0.7*(1.+0.1)**2, sm6['DSigma_err']*0.7*(1.+0.1)**2, capsize=5, fmt='bv', ms=10)
plt.errorbar(sm6['R']/1E3/0.7/(1.+sm6_z), sm6['DSigma']*0.7*(1.+sm6_z)**2., sm6['DSigma_err']*0.7*(1.+sm6_z)**2, capsize=5, mfc='None', ecolor='0.3', mec='0.3', fmt='^', ms=10)

plt.plot([0.32, 0.32], [6.0E2, 6.0E2], 'o', color='b', mec='b', ms=9)  
plt.plot([0.32, 0.32], [4.0E2, 4.0E2], '^', mfc='None', mec='0.3', ms=9)
plt.text(0.40, 5.5E2, r"\textbf{Mg II (z$\sim$0.5, Zhu et al. 2013)}", color='b', fontsize=14)
plt.text(0.40, 3.7E2, r"\textbf{Dark Matter (z$\sim$0.1, Mandelbaum et al. 2006)}", color='0.3', fontsize=14)
#plt.text(0.42, 4.8E2, r"\textbf{Mg II at $z\sim0.6$ (Zhu et al. 2013)}", color='b', fontsize=14)
#plt.text(0.42, 2.65E2, r"\textbf{Dark Matter at $z\sim 0.1$ (Mandelbaum et al. 2006)}", color='0.3', fontsize=14)
#plt.text([r"\textbf{Zhu, M\'enard \& SDSS (Mg II)}", r"\textbf{Mandelbaum \& SDSS (Dark Matter)}"], color='b', 
#plt.legend([r"\textbf{Zhu et al. (2013, Mg II)}", r"\textbf{Mandelbaum et al. (2006, Dark Matter)}"], frameon=False, loc=1)

plt.loglog(RR, DSigma_all_max, 'b', lw=3)
plt.loglog(RR, DSigma_all_max_1h, 'g', ls='--', dashes=dashes1, lw=3)
plt.loglog(RR, DSigma_all_max_2h, color='#FF8C00', ls='--', dashes=dashes2, lw=3) 

plt.legend([r"\textbf{Halo model}",r'\textbf{1-halo term}', r'\textbf{2-halo term}'], handlelength=3.5, frameon=False, loc=3)#, fontsize=14)
#plt.legend([r"\textbf{Zhu et al. (Mg II)}", r"\textbf{Mandelbaum et al. (Dark Matter)}", r"\textbf{Halo model}",r'\textbf{1-halo term}', r'\textbf{2-halo term}'], handlelength=3.5, frameon=False)#, fontsize=14)
#plt.gca().add_artist(l1)

#plt.errorbar(lf6['R']/1E3/0.7/(1.+lf6_z), lf6['DSigma']*0.7*(1.+lf6_z)**2., lf6['DSigma_err']*0.7*(1.+lf6_z)**2, capsize=5, fmt='c^', ms=10)
plt.xlabel(r'$\mathbf{r}_\mathrm{p}$', fontsize=24)
plt.ylabel(r'$\boldsymbol{\Delta \Sigma}\ (\mathrm{M}_\odot\ \mathrm{pc^{-2}})$', rotation='vertical', fontsize=26)
plt.xticks([0.1, 1, 10],('100 kpc', '1 Mpc', '10 Mpc'), fontsize=22)
plt.xlim(1.5E-2, 5E1)
plt.ylim(5E-1, 1E3)
#plt.text(1E-0, 5E2, r"\textbf{Zhu, M\'enard \& SDSS (Mg II)}", color='b', fontsize=12)
#plt.text(1E-0, 5E2, r"\textbf{Zhu \& SDSS (Mg II)}", color='b', fontsize=12)
#plt.text(1E-0, 3.5E2, r"\textbf{Mandelbaum \& SDSS (Dark Matter)}", color='0.5', fontsize=12)
plt.show()
plt.savefig('Darkeverything_sat1.eps')

show_lrg_err = np.zeros(Sigma_lrg.size)
for i in np.arange(Sigma_lrg.size): show_lrg_err[i] = min(DSigma_all_max_lrg[i]*0.9999, DSigma_all_max_lrg_err[i])
plt.figure(figsize=(9,13))
plt.clf()
ax1 = plt.subplot2grid((3,1),(0,0), rowspan=1)
plt.subplots_adjust(left=0.20, bottom=0.2, hspace=0)
plt.errorbar(y, Sigma_lrg, yerr=Sigma_lrg_err, capsize=5, fmt='bo', mec='b', ms=10)
plt.text(1.20, 1.5E3, r"\textbf{Mg II (z$\sim$0.5, Zhu et al. 2013)}", color='b', fontsize=14)
plt.xlim(2E-2, 3E1)
plt.ylim(5E-1, 4E3)
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\boldsymbol{\Sigma}_\mathrm{Mg\,II}\ (10^{-9} \mathrm{M}_\odot\ \mathrm{pc^{-2}})$', rotation='vertical', fontsize=24)
plt.title(r"Combining Galaxy-Gas/Mass Correlations (Demo)", fontsize=18)
plt.setp(ax1.get_xticklabels(), visible=False)

ax2 = plt.subplot2grid((3,1),(1,0), rowspan=1)
plt.loglog(RR, DSigma_all_sm6, 'b', lw=3)
plt.loglog(RR, DSigma_all_sm6_1h, 'g', ls='--', dashes=dashes1, lw=3)
plt.loglog(RR, DSigma_all_sm6_2h, color='#FF8C00', ls='--', dashes=dashes2, lw=3) 
plt.legend([r"\textbf{Halo model}",r'\textbf{1-halo term}', r'\textbf{2-halo term}'], handlelength=3.5, frameon=False, loc=3)
plt.errorbar(sm6['R']/1E3/0.7/(1.+sm6_z), sm6['DSigma']*0.7*(1.+sm6_z)**2., sm6['DSigma_err']*0.7*(1.+sm6_z)**2, capsize=5, mfc='None', ecolor='magenta', mec='magenta', fmt='^', ms=10)
plt.text(0.20, 7.0E2, r"\textbf{Dark Matter (z$\sim$0.2, Mandelbaum et al. 2006)}", color='magenta', fontsize=14)
plt.loglog(RR, DSigma_all_sm6, 'b', lw=3)
plt.loglog(RR, DSigma_all_sm6_1h, 'g', ls='--', dashes=dashes1, lw=3)
plt.loglog(RR, DSigma_all_sm6_2h, color='#FF8C00', ls='--', dashes=dashes2, lw=3) 
plt.ylabel(r'$\boldsymbol{\Delta \Sigma}_\mathrm{m}\ (\mathrm{M}_\odot\ \mathrm{pc^{-2}})$', rotation='vertical', fontsize=24)
plt.xlim(2E-2, 3E1)
plt.ylim(1E-1, 2E3)
plt.xscale('log')
plt.yscale('log')
plt.setp(ax2.get_xticklabels(), visible=False)

fSigma = interpolate.interp1d(RR, Sigma_all_sm6_highz)
Sigma_all_mgii = fSigma(y)
ax3 = plt.subplot2grid((3,1),(2,0), rowspan=1)
#plt.axhspan(A[iAmax_1h_sm6_left], A[iAmax_1h_sm6_right], ec='#90EE90', fc='#90EE90', lw=2)
#plt.axhspan(A[iAmax_2h_sm6_left], A[iAmax_2h_sm6_right], ec='#FFDAB9', fc='#FFDAB9', lw=2)
#plt.axhspan(A[iAmax_1h_sm6_left], A[iAmax_2h_sm6_right], ec='#E6E6FA', fc='#E6E6FA', lw=2)
#plt.axhspan(A[iAmax_1h_sm6_left], A[iAmax_1h_sm6_right], ec='green', facecolor='None', alpha=1.0, hatch='/', lw=2)
#plt.axhspan(A[iAmax_2h_sm6_left], A[iAmax_2h_sm6_right], ec='#FF8C00', facecolor='None', alpha=1.0, hatch='\\', lw=2)
#plt.axhspan((A[iAmax_1h_sm6_left]), (A[iAmax_1h_sm6_right]), facecolor='green', alpha=0.5)
#plt.axhspan((A[iAmax_2h_sm6_left]), (A[iAmax_2h_sm6_right]), facecolor='#FF8C00', alpha=0.5)
#plt.axhspan(np.log10(A[iAmax_1h_sm6_left]), np.log10(A[iAmax_1h_sm6_right]), facecolor='green', alpha=0.3)
#plt.axhspan(np.log10(A[iAmax_2h_sm6_left]), np.log10(A[iAmax_2h_sm6_right]), facecolor='#FF8C00', alpha=0.3)
plt.axhspan(2.8, 9.7, facecolor='blue', alpha=0.2)
#plt.legend([r'\textbf{1-halo term}', r'\textbf{2-halo term}'], handlelength=3.5, frameon=False, loc=2)
#ratio = np.log10(Sigma_lrg/Sigma_all_mgii)
#ratio_err = Sigma_lrg_err/Sigma_lrg/np.log(10.)
ratio = (Sigma_lrg/Sigma_all_mgii)
ratio_err = Sigma_lrg_err/Sigma_all_mgii
plt.errorbar(y, ratio, yerr=ratio_err, capsize=5, fmt='o', color='b', mec='b', ms=10)
plt.text(1.30, 16, r"\textbf{Mg II Gas-to-Mass Ratio}", color='b', fontsize=14)
plt.xlabel(r'$\mathbf{r}_\mathrm{p}$', fontsize=24)
#plt.ylabel(r'$\boldsymbol{\Delta \Sigma}\ (\mathrm{M}_\odot\ \mathrm{pc^{-2}})$', rotation='vertical', fontsize=26)
#plt.ylabel(r'$\log_{10}\ \boldsymbol{f}_\mathrm{Mg\,II}/10^{-9}$', rotation='vertical', fontsize=24)
plt.ylabel(r'$\boldsymbol{f}_\mathrm{Mg\,II}/10^{-9}$', rotation='vertical', fontsize=24)
plt.xlim(2E-2, 3E1)
#plt.ylim(0.00, 1.79)
#plt.ylim(10**0.20, 10**2.09)
plt.ylim(-10**0.5, 19.)
plt.xscale('log')
plt.xticks([0.1, 1, 10],('100 kpc', '1 Mpc', '10 Mpc'), fontsize=22)

plt.show()
plt.savefig('Darkeverything_Demo_sat1.pdf')
plt.savefig('Darkeverything_Demo_sat1.eps')

plt.figure(figsize=(10,7))
plt.subplots_adjust(left=0.20, bottom=0.2)
show_lrg_err = np.zeros(Sigma_lrg.size)
for i in np.arange(Sigma_lrg.size): show_lrg_err[i] = min(DSigma_all_max_lrg[i]*0.9999, DSigma_all_max_lrg_err[i])

plt.loglog(RR, DSigma_all_max, 'b', lw=3)
plt.loglog(RR, DSigma_all_max_1h, 'g', ls='--', dashes=dashes1, lw=3)
plt.loglog(RR, DSigma_all_max_2h, color='#FF8C00', ls='--', dashes=dashes2, lw=3) 
plt.legend([r"\textbf{Halo model}",r'\textbf{1-halo term}', r'\textbf{2-halo term}'], handlelength=3.5, frameon=False, loc=3)#, fontsize=14)

plt.errorbar(y, DSigma_all_max_lrg, yerr=show_lrg_err, capsize=5, fmt='o', color='b', mec='b', ms=10)
#plt.errorbar(sm6['R']/1E3/0.7/(1.+0.1), sm6['DSigma']*0.7*(1.+0.1)**2, sm6['DSigma_err']*0.7*(1.+0.1)**2, capsize=5, fmt='bv', ms=10)
plt.errorbar(sm6['R']/1E3/0.7/(1.+sm6_z), sm6['DSigma']*0.7*(1.+sm6_z)**2., sm6['DSigma_err']*0.7*(1.+sm6_z)**2, capsize=5, mfc='None', ecolor='0.3', mec='0.3', fmt='^', ms=10)

plt.plot([0.32, 0.32], [6.0E2, 6.0E2], 'o', color='b', mec='b', ms=9)  
plt.plot([0.32, 0.32], [4.0E2, 4.0E2], '^', mfc='None', mec='0.3', ms=9)
plt.text(0.40, 5.5E2, r"\textbf{Mg II (z$\sim$0.6, Zhu et al. 2013)}", color='b', fontsize=14)
plt.text(0.40, 3.7E2, r"\textbf{Dark Matter (z$\sim$0.1, Mandelbaum et al. 2006)}", color='0.3', fontsize=14)
#plt.text(0.42, 4.8E2, r"\textbf{Mg II at $z\sim0.6$ (Zhu et al. 2013)}", color='b', fontsize=14)
#plt.text(0.42, 2.65E2, r"\textbf{Dark Matter at $z\sim 0.1$ (Mandelbaum et al. 2006)}", color='0.3', fontsize=14)
#plt.text([r"\textbf{Zhu, M\'enard \& SDSS (Mg II)}", r"\textbf{Mandelbaum \& SDSS (Dark Matter)}"], color='b', 
#plt.legend([r"\textbf{Zhu et al. (2013, Mg II)}", r"\textbf{Mandelbaum et al. (2006, Dark Matter)}"], frameon=False, loc=1)

plt.loglog(RR, DSigma_all_max, 'b', lw=3)
plt.loglog(RR, DSigma_all_max_1h, 'g', ls='--', dashes=dashes1, lw=3)
plt.loglog(RR, DSigma_all_max_2h, color='#FF8C00', ls='--', dashes=dashes2, lw=3) 

plt.legend([r"\textbf{Halo model}",r'\textbf{1-halo term}', r'\textbf{2-halo term}'], handlelength=3.5, frameon=False, loc=3)#, fontsize=14)
#plt.legend([r"\textbf{Zhu et al. (Mg II)}", r"\textbf{Mandelbaum et al. (Dark Matter)}", r"\textbf{Halo model}",r'\textbf{1-halo term}', r'\textbf{2-halo term}'], handlelength=3.5, frameon=False)#, fontsize=14)
#plt.gca().add_artist(l1)

#plt.errorbar(lf6['R']/1E3/0.7/(1.+lf6_z), lf6['DSigma']*0.7*(1.+lf6_z)**2., lf6['DSigma_err']*0.7*(1.+lf6_z)**2, capsize=5, fmt='c^', ms=10)
plt.xlabel(r'$\mathbf{r}_\mathrm{p}$', fontsize=24)
plt.ylabel(r'$\boldsymbol{\Delta \Sigma}\ (\mathrm{M}_\odot\ \mathrm{pc^{-2}})$', rotation='vertical', fontsize=26)
plt.xticks([0.1, 1, 10],('100 kpc', '1 Mpc', '10 Mpc'), fontsize=22)
plt.xlim(1.5E-2, 5E1)
plt.ylim(5E-1, 1E3)
#plt.text(1E-0, 5E2, r"\textbf{Zhu, M\'enard \& SDSS (Mg II)}", color='b', fontsize=12)
#plt.text(1E-0, 5E2, r"\textbf{Zhu \& SDSS (Mg II)}", color='b', fontsize=12)
#plt.text(1E-0, 3.5E2, r"\textbf{Mandelbaum \& SDSS (Dark Matter)}", color='0.5', fontsize=12)
plt.show()
plt.savefig('Darkeverything_sat1.eps')


plt.figure(figsize=(10,10))
plt.clf()
ax1 = plt.subplot2grid((4,1),(0,0), rowspan=3)
plt.subplots_adjust(left=0.15, bottom=0.15, hspace=0)
plt.loglog(RR, Amax_1h*Sigma_1h_max+Amax_2h*Sigma_2h_max, 'b', lw=3)
plt.loglog(RR, Amax_1h*Sigma_1h_max, 'g', ls='--', dashes=dashes1, lw=3, label=r'\textbf{1-halo term}')
plt.loglog(RR, Amax_2h*Sigma_2h_max, color='#FF8C00', ls='--', dashes=dashes2, lw=3, label=r'\textbf{2-halo term}')
plt.legend([r'\textbf{Halo model}', r'\textbf{1-halo term}', r'\textbf{2-halo term}'], handlelength=3.4, frameon=False, loc=3)#, fontsize=14)
plt.errorbar(y, Sigma_lrg, yerr=Sigma_lrg_err, capsize=5, fmt='bo', mec='b', ms=10)
#plt.errorbar(y[0:4], Sigma_lrg[0:4]/1.2, yerr=Sigma_lrg_err[0:4]/1.2, capsize=5, fmt='bo', mec='b', ms=10, mfc=None)
plt.ylabel(r'$\boldsymbol{\Sigma}_\mathrm{Mg\,II}\ (10^{-8} \mathrm{M}_\odot\ \mathrm{pc^{-2}})$', rotation='vertical', fontsize=26)
#plt.xlim(1E-2, 4E1)
plt.xlim(2E-2, 3E1)
#plt.ylim(5E-1, 4E3)
plt.ylim(2E-1, 5E2)
#plt.text(3E-2, 3E-1, r'$\frac{\boldsymbol{\chi}^2}{\textbf{dof}}='+'%4.2f' % (Chi2[iAmax_1h_all, iAmax_2h_all, iMmax_all]/14.)+'$', fontsize=22)
plt.text(1.5E0, 19E1, r'$\log_{10} \boldsymbol{\mathrm{M}}_\mathrm{halo}/\boldsymbol{\mathrm{M}}_\odot='+'%4.1f' % (np.log10(Mhalo[iMmax_all])+0.000)+'$', fontsize=20)
plt.text(1.5E0, 10E1, r'$\boldsymbol{\chi}^2/{dof}='+'%4.2f' % (Chi2[iAmax_1h_all, iAmax_2h_all, iMmax_all]/14.)+'$', fontsize=20)
#plt.text(1.5E0, 9E2, r'$\boldsymbol{\chi}^2/{dof}='+'%4.2f' % (Chi2[iAmax_1h_all, iAmax_2h_all, iMmax_all]/14.)+'$', fontsize=20)
plt.title(r"Saturation Effects (line ratio=1)", fontsize=18)

ax2 = plt.subplot2grid((4,1),(3,0), rowspan=1)
#plt.errorbar(y, Sigma_lrg/Sigma_all[:,iAmax_1h_all,iAmax_2h_all,iMmax_all], yerr=Sigma_lrg_err/Sigma_all[:,iAmax_1h_all,iAmax_2h_all,iMmax_all], capsize=5, fmt='bo', mec='b', ms=10)
plt.errorbar(y, (Sigma_lrg-Sigma_all[:,iAmax_1h_all,iAmax_2h_all,iMmax_all])/Sigma_all[:,iAmax_1h_all,iAmax_2h_all,iMmax_all], yerr=Sigma_lrg_err/Sigma_all[:,iAmax_1h_all,iAmax_2h_all,iMmax_all], capsize=5, fmt='bo', mec='b', ms=10)
plt.xscale('log')
plt.plot([1E-2,5E1],[0,0], 'k--', lw=1)
plt.xticks([0.1, 1, 10],('100 kpc', '1 Mpc', '10 Mpc'), fontsize=22)
plt.yticks([-1., 0., 1])
plt.xlim(2E-2, 3E1)
plt.ylim(-1.99E0, 1.99E0)
plt.xlabel(r'$\mathbf{r}_\mathrm{p}$', fontsize=24)
plt.ylabel(r'$\boldsymbol{\delta \Sigma}_\mathrm{Mg\,II}/\boldsymbol{\Sigma}_\mathrm{Mg\,II}^\mathrm{model}$', rotation='vertical', fontsize=26)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.show()
plt.savefig('Halomodel_residuals_sat1.eps')

plt.figure(figsize=(20,10))
plt.clf()
ax11 = plt.subplot2grid((4,2),(0,0), rowspan=3)
plt.loglog(RR, Amax_1h_12*Sigma_1h_12+Amax_2h_12*Sigma_2h_12, color='r', lw=4)
plt.loglog(RR, Amax_1h_12*Sigma_1h_12, color='m', ls='--', dashes=dashes1, lw=4, label=r'\textbf{1-halo term}')
plt.loglog(RR, Amax_2h_12*Sigma_2h_12, color='m', ls='--', dashes=dashes2, lw=4, label=r'\textbf{2-halo term}')
plt.legend([r'\textbf{Halo model}', r'\textbf{1-halo term}', r'\textbf{2-halo term}'], handlelength=3.4, frameon=False, loc=3)#, fontsize=14)
plt.errorbar(y, Sigma_lrg, yerr=Sigma_lrg_err, capsize=5, fmt='bo', mec='b', ms=10)
plt.ylabel(r'$\boldsymbol{\Sigma}_\mathrm{Mg\,II}\ (10^{-9} \mathrm{M}_\odot\ \mathrm{pc^{-2}})$', rotation='vertical', fontsize=26)
plt.xlim(2E-2, 3E1)
plt.ylim(5E-1, 4E3)
plt.text(0.8E0, 17E2, r'$\log_{10} \boldsymbol{\mathrm{M}}_\mathrm{halo}/\boldsymbol{\mathrm{M}}_\odot='+'%4.1f' % np.log10(Mhalo[0])+'$', fontsize=20)
plt.text(0.8E0, 9E2, r'$\boldsymbol{\chi}^2/{dof}='+'%4.2f' % (Chi2[iAmax_1h_12, iAmax_2h_12, 0]/14.)+'$', fontsize=20)
plt.setp(ax11.get_xticklabels(), visible=False)

ax22 = plt.subplot2grid((4,2),(3,0), rowspan=1)
plt.errorbar(y, (Sigma_lrg-Sigma_all[:,iAmax_1h_12,iAmax_2h_12,0])/Sigma_all[:,iAmax_1h_12,iAmax_2h_12,0], yerr=Sigma_lrg_err/Sigma_all[:,iAmax_1h_12,iAmax_2h_12,0], capsize=5, fmt='bo', mec='b', ms=10)
plt.xscale('log')
plt.plot([1E-2,5E1],[0,0], 'k--', lw=1)
plt.xticks([0.1, 1, 10],('100 kpc', '1 Mpc', '10 Mpc'), fontsize=22)
plt.yticks([-1., 0., 1])
plt.xlim(2E-2, 3E1)
plt.ylim(-1.99E0, 1.99E0)
plt.xlabel(r'$\mathbf{r}_\mathrm{p}$', fontsize=24)
plt.ylabel(r'$\boldsymbol{\delta \Sigma}_\mathrm{Mg\,II}/\boldsymbol{\Sigma}_\mathrm{Mg\,II}^\mathrm{model}$', rotation='vertical', fontsize=26)

ax33 = plt.subplot2grid((4,2),(0,1), rowspan=3)
plt.loglog(RR, Amax_1h_15*Sigma_1h_15+Amax_2h_15*Sigma_2h_15, color='#D2691E', lw=4)
plt.loglog(RR, Amax_1h_15*Sigma_1h_15, color='#FF8C00', ls='--', dashes=dashes1, lw=4, label=r'\textbf{1-halo term}')
plt.loglog(RR, Amax_2h_15*Sigma_2h_15, color='#FF8C00', ls='--', dashes=dashes2, lw=4, label=r'\textbf{2-halo term}')
plt.legend([r'\textbf{Halo model}', r'\textbf{1-halo term}', r'\textbf{2-halo term}'], handlelength=3.4, frameon=False, loc=3)#, fontsize=14)
plt.errorbar(y, Sigma_lrg, yerr=Sigma_lrg_err, capsize=5, fmt='bo', mec='b', ms=10)
plt.xlim(2E-2, 3E1)
plt.ylim(5E-1, 4E3)
plt.text(0.8E0, 17E2, r'$\log_{10} \boldsymbol{\mathrm{M}}_\mathrm{halo}/\boldsymbol{\mathrm{M}}_\odot='+'%4.1f' % np.log10(Mhalo[nMhalo-1])+'$', fontsize=20)
plt.text(0.8E0, 9E2, r'$\boldsymbol{\chi}^2/{dof}='+'%4.2f' % (Chi2[iAmax_1h_15, iAmax_2h_15, nMhalo-1]/14.)+'$', fontsize=20)
plt.ylabel(r'$\boldsymbol{\Sigma}_\mathrm{Mg\,II}\ (10^{-9} \mathrm{M}_\odot\ \mathrm{pc^{-2}})$', rotation='vertical', fontsize=26)
ax33.yaxis.tick_right()
ax33.yaxis.set_label_position("right")
plt.setp(ax33.get_xticklabels(), visible=False)
#plt.setp(ax33.get_yticklabels(), visible=False)

ax44 = plt.subplot2grid((4,2),(3,1), rowspan=1)
plt.errorbar(y, (Sigma_lrg-Sigma_all[:,iAmax_1h_15,iAmax_2h_15,nMhalo-1])/Sigma_all[:,iAmax_1h_15,iAmax_2h_15,nMhalo-1], yerr=Sigma_lrg_err/Sigma_all[:,iAmax_1h_15,iAmax_2h_15,nMhalo-1], capsize=5, fmt='bo', mec='b', ms=10)
plt.xscale('log')
plt.plot([1E-2,5E1],[0,0], 'k--', lw=1)
plt.xticks([0.1, 1, 10],('100 kpc', '1 Mpc', '10 Mpc'), fontsize=22)
plt.yticks([-1., 0., 1])
plt.xlim(2E-2, 3E1)
plt.ylim(-1.99E0, 1.99E0)
ax44.yaxis.tick_right()
plt.xlabel(r'$\mathbf{r}_\mathrm{p}$', fontsize=24)
plt.ylabel(r'$\boldsymbol{\delta \Sigma}_\mathrm{Mg\,II}/\boldsymbol{\Sigma}_\mathrm{Mg\,II}^\mathrm{model}$', rotation='vertical', fontsize=26)
ax44.yaxis.set_label_position("right")
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.85, hspace=0, wspace=0)
plt.show()
plt.savefig('Halomodel_comparison_residuals_extremes_sat1.eps')

plt.figure(figsize=(9,6))
plt.clf()
plt.subplots_adjust(left=0.20, bottom=0.15, hspace=0)
cs=plt.contour(np.log10(Mhalo[0:nMhalo]), np.log10(A[0:nA]), joint_2h_Mhalo[0:nA, 0:nMhalo], levels_2h_Mhalo, colors=('#FF8C00','#FF8C00','#FF8C00'))
cs=plt.contour(np.log10(Mhalo[0:nMhalo]), np.log10(A[0:nA]), joint_1h_Mhalo[0:nA, 0:nMhalo], levels_1h_Mhalo, colors=('g', 'g', 'g'))
#plt.text(12, -0.5, r'$\boldsymbol{f}_\mathrm{Mg\,II}^\mathrm{1h}=\Sigma_mathrm{Mg\,II}^\mathrm{1h}/\Sigma_mathrm{m}^\mathrm{1h} (1-halo gas-to-mass ratio)$', color='b', fontsize=20)
#plt.text(12, -0.8, r'$\boldsymbol{f}_\mathrm{Mg\,II}^\mathrm{2h}=\Sigma_mathrm{Mg\,II}^\mathrm{2h}/\Sigma_mathrm{m}^\mathrm{2h} (2-halo gas-to-mass ratio)$', color='b', fontsize=20)
plt.text(13.8, 1.70, r'green: 1-halo gas-to-mass ratio', color='g', fontsize=16)
plt.text(13.8, 1.53, r'orange: 2-halo gas-to-mass ratio', color='#FF8C00', fontsize=16)
plt.xlim(np.log10(Mhalo_min), 16.5)
plt.ylim(-1.2, 1.99)
plt.xlabel(r'$\mathbf{\log}_{10}\ \mathbf{M}_\mathrm{halo}/\mathbf{M}_\mathrm{\odot}$', fontsize=24)
plt.ylabel(r'$\log_{10}\ \boldsymbol{f}_\mathrm{Mg\,II}^\mathrm{1h, 2h}/10^{-9}$', rotation='vertical', fontsize=30)
plt.plot(np.log10(Mhalo[iMmax_1h]), np.log10(A[iAmax_1h]), 'o', color='green', markersize=15)
plt.plot(np.log10(Mhalo[iMmax_2h]), np.log10(A[iAmax_2h]), 'o', color='#FF8C00', markersize=15)
plt.savefig('Chi2_1h_2h_onepanel_sat1.eps')

plt.figure(figsize=(9,6))
plt.clf()
plt.subplots_adjust(left=0.20, bottom=0.15, hspace=0)
cs=plt.contour(np.log10(Mhalo[0:nMhalo]), np.log10(A[0:nA]), joint_2h_Mhalo[0:nA, 0:nMhalo], levels_2h_Mhalo, colors=('#FF8C00','#FF8C00','#FF8C00'))
cs=plt.contour(np.log10(Mhalo[0:nMhalo]), np.log10(A[0:nA]), joint_1h_Mhalo[0:nA, 0:nMhalo], levels_1h_Mhalo, colors=('g', 'g', 'g'))
#plt.text(12, -0.5, r'$\boldsymbol{f}_\mathrm{Mg\,II}^\mathrm{1h}=\Sigma_mathrm{Mg\,II}^\mathrm{1h}/\Sigma_mathrm{m}^\mathrm{1h} (1-halo gas-to-mass ratio)$', color='b', fontsize=20)
#plt.text(12, -0.8, r'$\boldsymbol{f}_\mathrm{Mg\,II}^\mathrm{2h}=\Sigma_mathrm{Mg\,II}^\mathrm{2h}/\Sigma_mathrm{m}^\mathrm{2h} (2-halo gas-to-mass ratio)$', color='b', fontsize=20)
plt.xlim(10.9, 16.0)
plt.ylim(-1.69, 1.5)
plt.xlabel(r'$\mathbf{\log}_{10}\ \mathbf{M}_\mathrm{halo}/\mathbf{M}_\mathrm{\odot}$', fontsize=24)
#plt.ylabel(r'$\log_{10}\ \boldsymbol{f}_\mathrm{Mg\,II}^\mathrm{1h, 2h}/10^{-8}$', rotation='vertical', fontsize=30)
plt.ylabel(r'$\log_{10}\ \boldsymbol{f}_\mathrm{Mg\,II}/10^{-8}$', rotation='vertical', fontsize=30)
plt.plot(np.log10(Mhalo[iMmax_1h]), np.log10(A[iAmax_1h]), 'o', color='green', markersize=8)
plt.plot(np.log10(Mhalo[iMmax_2h]), np.log10(A[iAmax_2h]), 'o', color='#FF8C00', markersize=8)
plt.axhspan(np.log10(0.6), np.log10(1.26), facecolor='blue', alpha=0.2)
plt.axvspan(13.25, 13.45, facecolor='0.8', alpha=0.3)
#plt.axhspan(np.log10(A[iAmax_1h_sm6_left]), np.log10(A[iAmax_1h_sm6_right]), facecolor='green', alpha=0.3)
#plt.title(r"Combining Galaxy-Gas/Mass Correlations", fontsize=18)
plt.text(13.0, 1.33, r'green: 1-halo gas-to-mass ratio', color='g', fontsize=16)
plt.text(13.0, 1.20, r'orange: 2-halo gas-to-mass ratio', color='#FF8C00', fontsize=16)
plt.text(11.14, -1.35, r'gray band: mass constraint from galaxy-mass correlation', color='0.3', fontsize=13)
plt.text(11.14, -1.5, r'blue band: gas-to-mass ratio constraint from galaxy-gas/mass correlation', color='blue', fontsize=13)
plt.title(r"Saturation Effects (line ratio=1)", fontsize=18)
plt.savefig('Chi2_1h_2h_onepanel_dark_sat1.pdf')
plt.savefig('Chi2_1h_2h_onepanel_dark_sat1.eps')

plt.figure(figsize=(9,12))
plt.clf()
ax3 = plt.subplot2grid((2,1),(0,0), rowspan=1)
plt.axhspan(0.70, 1.20, facecolor='0.9', alpha=0.3)
plt.axvspan(13.25, 13.45, facecolor='0.9', alpha=0.3)
plt.subplots_adjust(left=0.20, bottom=0.15, hspace=0)
csf=plt.contourf(np.log10(Mhalo[0:nMhalo]), np.log10(A[0:nA]), joint_1h_Mhalo[0:nA, 0:nMhalo], levels_1h_Mhalo_f[::-1], colors=('white','white','blue'))
cs=plt.contour(np.log10(Mhalo[0:nMhalo]), np.log10(A[0:nA]), joint_1h_Mhalo[0:nA, 0:nMhalo], levels_1h_Mhalo, colors=('k', 'k', 'k'))
#plt.xlim(12, 16)
plt.xlim(np.log10(Mhalo_min), 16.5)
#plt.xlim(np.log10(Mhalo_min), np.log10(Mhalo_max))
#plt.ylim(np.log10(A_min), np.log10(A_max))
plt.ylim(-1.2, 1.99)
# plt.xlabel(r'$\mathbf{\log}_{10}\ \mathbf{M}_\mathrm{halo}/\mathbf{M}_\mathrm{\odot}$', fontsize=24)
# plt.ylabel(r'$\bigg(\frac{\mathbf{\Sigma}_\mathrm{Mg\,II}}{\mathbf{\Sigma}_\mathrm{m}}\bigg)^\mathrm{1h}$', rotation='horizontal', fontsize=30)
plt.ylabel(r'$\log_{10}\ \boldsymbol{f}_\mathrm{Mg\,II}^\mathrm{1h}/10^{-9}$', rotation='vertical', fontsize=30)
strs =[r'$68.3\%$', r'$95.4\%$', r'$99.7\%$']
fmt = {}
for l,s in zip(cs.levels, strs ): fmt[l] = s
#plt.clabel(cs, cs.levels, fmt=fmt, inline=1, fontsize=10)
#plt.plot(np.log10(Mhalo[iMmax_1h]), np.log10(A[iAmax_1h]), 'o', color='white', markersize=12)
plt.plot(np.log10(Mhalo[iMmax_1h]), np.log10(A[iAmax_1h]), 'o', color='yellow', markersize=15)
plt.title(r"Combining Galaxy-Gas/Mass Correlations (Demo)", fontsize=18)

ax4 = plt.subplot2grid((2,1),(1,0), rowspan=1)
plt.axhspan(0.60, 1.10, facecolor='0.9', alpha=0.3)
plt.axvspan(13.25, 13.45, facecolor='0.9', alpha=0.3)
csf=plt.contourf(np.log10(Mhalo[0:nMhalo]), np.log10(A[0:nA]), joint_2h_Mhalo[0:nA, 0:nMhalo], levels_2h_Mhalo_f[::-1], colors=('white', 'white', 'blue'))
cs=plt.contour(np.log10(Mhalo[0:nMhalo]), np.log10(A[0:nA]), joint_2h_Mhalo[0:nA, 0:nMhalo], levels_2h_Mhalo, colors=('k','k','k'))
#plt.xlim(12, 16.)
plt.xlim(np.log10(Mhalo_min), 16.5)
#plt.xlim(np.log10(Mhalo_min), np.log10(Mhalo_max))
#plt.ylim(np.log10(A_min), np.log10(A_max))
plt.ylim(-1.2, 1.99)
plt.xlabel(r'$\mathbf{\log}_{10}\ \mathbf{M}_\mathrm{halo}/\mathbf{M}_\mathrm{\odot}$', fontsize=24)
# plt.ylabel(r'$\bigg(\frac{\mathbf{\Sigma}_\mathrm{Mg\,II}}{\mathbf{\Sigma}_\mathrm{m}}\bigg)^\mathrm{2h}$', rotation='horizontal', fontsize=30)
plt.ylabel(r'$\log_{10}\ \boldsymbol{f}_\mathrm{Mg\,II}^\mathrm{2h}/10^{-9}$', rotation='vertical', fontsize=30)
strs =[r'$68.3\%$', r'$95.4\%$', r'$99.7\%$']
fmt = {}
for l,s in zip(cs.levels, strs ): fmt[l] = s
#plt.clabel(cs, cs.levels, fmt=fmt, inline=1, fontsize=10)
plt.plot(np.log10(Mhalo[iMmax_2h]), np.log10(A[iAmax_2h]), 'o', color='yellow', markersize=15)

#plt.plot(np.log10(Mhalo[iMmax_2h]), np.log10(A[iAmax_2h]), 'bx', markersize=15)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.show()
plt.savefig('Chi2_1h_2h_sat1.eps')

#print "velocity dispersion ..."
#plt.figure(figsize=(10,7))
#plt.clf()
#ax = plt.subplot2grid((1,1),(0,0), rowspan=1)
#plt.subplots_adjust(left=0.20, bottom=0.2)
#plt.loglog(RR, vdisp_all_max, 'b', lw=2)
#plt.loglog(RR, vdisp_1h_max, 'g', ls='--', dashes=dashes1, lw=2)
#plt.loglog(RR, vdisp_2h_max, color='#FF8C00', ls='--', dashes=dashes2, lw=2)
#plt.loglog(RR, vdisp_1h_max_dm, 'gray', ls=':', lw=2)
#
#plt.legend([r'\textbf{Halo model}', r'\textbf{1-halo term (gas)}', r'\textbf{2-halo term (gas\&dark matter)}', r'\textbf{1-halo term (dark matter)}'], handlelength=3.4, frameon=False, loc=3)
#plt.errorbar(y, vdisp_lrg, yerr=vdisp_lrg_err, capsize=5, fmt='bo', mec='b', ms=10)
#plt.setp(ax.get_yticklabels(), visible=True)
#plt.yticks([20, 50, 100, 200, 400],('20', '50', '100', '200', '400'), fontsize=22)
#plt.ylabel(r'$\boldsymbol{\sigma}_\mathrm{los}\ (\mathrm{km\ s^{-1}})$', rotation='vertical', fontsize=26)
#plt.setp(ax.get_xticklabels(), visible=True)
#plt.xticks([0.1, 1, 10],('100 kpc', '1 Mpc', '10 Mpc'), fontsize=22)
#plt.xlabel(r'$\mathbf{r}_\mathrm{p}$', fontsize=26)
#plt.xlim(1E-2, 3E1)
#plt.ylim(1.0E1, 6.5E2)
#plt.show()
#plt.savefig('Halomodel_vdisp_sat1.eps')

print "Demo running ..."
ndemo = 3
#M_demo = np.arange(8)*0.5+12.0
#M_demo = np.array([11.5, 12.5, 13.5, 14.5, 15.5])
#outcolor=[plt.cm.hsv(0.01), plt.cm.hsv(0.1), plt.cm.hsv(0.7), plt.cm.hsv(0.85), plt.cm.hsv(0.95)]
#dashes = [(8,3), (11,4), (14,5), (17, 6), (20,7)]
#M_demo = np.array([11.5, 13.4, 15.5])
M_demo = np.array([12.0, np.log10(Mmax), 16.0])
outcolor=[plt.cm.hsv(0.01), plt.cm.hsv(0.7), plt.cm.hsv(0.1)]
dashes = [(10,3), (17, 6), (25,9)]

Sigma_1h_demo = np.zeros((RR.size, ndemo))
Sigma_2h_demo = np.zeros((RR.size, ndemo))
Amax_1h_demo = np.zeros(ndemo)
Amax_2h_demo = np.zeros(ndemo)
iAmax_1h_demo = np.zeros(ndemo)
iAmax_2h_demo = np.zeros(ndemo)
iMmax_demo = np.zeros(ndemo)
for i in np.arange(ndemo):
    Mtmp = M_demo[i]
    iMtmp = np.argmin(np.abs(Mtmp-np.log10(Mhalo)))
    iMmax_demo[i] = iMtmp
    bMtmp = bM[iMtmp]
    Sigma_1h_demo[:,i] = (hm.NFW_project_profile(RR, 10.**Mtmp, z, CosPar)).reshape(RR.size)
    Sigma_2h_demo[:,i] = (bMtmp*f(RR).reshape(RR.size,1)).reshape(RR.size)
    iAmax_1h_tmp = np.argmin(Chi2[:,:,iMtmp])/(nA)
    iAmax_2h_tmp = np.argmin(Chi2[:,:,iMtmp])%(nA)
    iAmax_1h_demo[i] = iAmax_1h_tmp
    iAmax_2h_demo[i] = iAmax_2h_tmp
    Amax_1h_demo[i] = A[iAmax_1h_tmp]
    Amax_2h_demo[i] = A[iAmax_2h_tmp]
    print iMtmp, iAmax_1h_tmp, iAmax_2h_tmp, nA, nMhalo

plt.figure(figsize=(10,7))
plt.clf()
plt.subplots_adjust(left=0.20, bottom=0.2)
#outcolor=[plt.cm.hsv(0.01), plt.cm.hsv(0.05), plt.cm.hsv(0.1), plt.cm.hsv(0.6), plt.cm.hsv(0.65), plt.cm.hsv(0.75), plt.cm.hsv(0.85), plt.cm.hsv(0.90)]
plt.xlim(1.5E-2, 4E1)
plt.ylim(4E-1, 6E3)
plt.errorbar(y, Sigma_lrg, yerr=Sigma_lrg_err, capsize=5, fmt='bo', mec='b', ms=10)
plt.text(1.3E0, 17E2, r'$\log_{10} \boldsymbol{\mathrm{M}}_\mathrm{halo}/\boldsymbol{\mathrm{M}}_\odot=$', color='k', fontsize=20)
for i in np.arange(ndemo):
    plt.loglog(RR, Amax_2h*(Sigma_1h_demo[:,i]+Sigma_2h_demo[:,i]), color=outcolor[i], lw=3)
    ytmp = 18.5E2/(1.8**(ndemo-1-i))
    plt.text(1.3E1, ytmp, r'$'+'%4.1f' % (M_demo[i]+0.000) +'$', color=outcolor[i], weight='bold', fontsize=20)

plt.xticks([0.1, 1, 10],('100 kpc', '1 Mpc', '10 Mpc'), fontsize=22)
plt.xlabel(r'$\mathbf{r}_\mathrm{p}$', fontsize=24)
plt.ylabel(r'$\boldsymbol{\Sigma}_\mathrm{Mg\,II}^\mathrm{model}\ (10^{-9} \mathrm{M}_\odot\ \mathrm{pc^{-2}})$', rotation='vertical', fontsize=26)
plt.show()
plt.savefig('Halomodel_demo_sat1.eps')

plt.figure(figsize=(9,13))
plt.clf()
ax = plt.subplot2grid((ndemo+3,1),(0,0), rowspan=3)
plt.subplots_adjust(left=0.15, bottom=0.15, hspace=0)
plt.text(1.3E0, 17E2, r'$\log_{10} \boldsymbol{\mathrm{M}}_\mathrm{halo}/\boldsymbol{\mathrm{M}}_\odot=$', color='k', fontsize=20)
for i in np.arange(ndemo):
    plt.loglog(RR, Amax_1h_demo[i]*Sigma_1h_demo[:,i]+Amax_2h_demo[i]*Sigma_2h_demo[:,i], color=outcolor[i], lw=4)
    #plt.loglog(RR, Amax_1h_demo[i]*Sigma_1h_demo[:,i]+Amax_2h_demo[i]*Sigma_2h_demo[:,i], '--', color=outcolor[i], dashes=dashes[i], lw=4)
    plt.loglog(RR, Amax_1h_demo[i]*Sigma_1h_demo[:,i], '--', color=outcolor[i], dashes=dashes1, lw=1)
    plt.loglog(RR, Amax_2h_demo[i]*Sigma_2h_demo[:,i], '--', color=outcolor[i], dashes=dashes2, lw=1)
    ytmp = 18.5E2/(1.8**i)
    plt.text(1.8E1, ytmp, r'$'+'%4.1f' % (M_demo[i]+0.000) +'$', color=outcolor[i], weight='heavy', fontsize=20)
#plt.xlabel(r'$\mathbf{r}_\mathrm{p}$', fontsize=24)
plt.ylabel(r'$\boldsymbol{\Sigma}_\mathrm{Mg\,II}\ (10^{-9} \mathrm{M}_\odot\ \mathrm{pc^{-2}})$', rotation='vertical', fontsize=26)
#plt.xlim(2E-2, 3E1)
#plt.ylim(5E-1, 4E3)
plt.xlim(1.5E-2, 5E1)
plt.ylim(5E-1, 4E3)
plt.setp(ax.get_xticklabels(), visible=False)
#plt.xticks([0.1, 1, 10],('100 kpc', '1 Mpc', '10 Mpc'), fontsize=22)
plt.errorbar(y, Sigma_lrg, yerr=Sigma_lrg_err, capsize=5, fmt='bo', mec='b', ms=14)
#plt.savefig('Halomodel_demo_bestfit.eps')

#plt.clf()
#plt.figure(figsize=(10,10))
#plt.subplots_adjust(left=0.15, bottom=0.15, hspace=0)
for i in np.arange(ndemo):
    ax = plt.subplot2grid((ndemo+3,1),(i+3,0), rowspan=1)
    plt.errorbar(y, (Sigma_lrg-Sigma_all[:,iAmax_1h_demo[i],iAmax_2h_demo[i],iMmax_demo[i]])/Sigma_all[:,iAmax_1h_demo[i],iAmax_2h_demo[i],iMmax_demo[i]], yerr=Sigma_lrg_err/Sigma_all[:,iAmax_1h_demo[i],iAmax_2h_demo[i],iMmax_demo[i]], capsize=5, fmt='o', mec=outcolor[i], mfc=outcolor[i], ecolor=outcolor[i], ms=10)
    #plt.plot(RR, ((Amax_1h*Sigma_1h_max+Amax_2h*Sigma_2h_max)-(Amax_1h_demo[i]*Sigma_1h_demo[:,i]+Amax_2h_demo[i]*Sigma_2h_demo[:,i])), '--', color=outcolor[i], dashes=dashes[i], lw=4)
    plt.xscale('log')
    plt.plot([1E-2,5E1],[0,0], 'k--', lw=1)
    plt.xlim(1.5E-2, 5E1)
    #if i != ndemo-1: plt.ylim(-1.49E0, 1.49E0)
    #if i == ndemo-1: plt.ylim(-1.E0, 2.E0)
    plt.ylim(-1.01E0, 1.99E0)
    plt.setp(ax.get_xticklabels(), visible=False)
    #if i == 0: plt.text(2E0, 1.2E0, r'$\log_{10} \boldsymbol{\mathrm{M}}_\mathrm{halo}/\boldsymbol{\mathrm{M}}_\odot='+'%4.1f' % M_demo[i] +'$', color=outcolor[i], weight='bold', fontsize=20)
    #if i != 0: plt.text(1.7E1, 1.2E0, r'$'+'%4.1f' % M_demo[i] +'$', color=outcolor[i], weight='bold', fontsize=20)
    plt.text(5E0, 1.25E0, r'$\boldsymbol{\chi}^2/{dof}='+'%4.2f' % (Chi2[iAmax_1h_demo[i], iAmax_2h_demo[i], iMmax_demo[i]]/14.)+'$', color=outcolor[i], fontsize=20)
    #if i == 0: plt.text(5E0, 1.0E0, r'$\boldsymbol{\chi}^2/{dof}='+'%4.2f' % (Chi2[iAmax_1h_demo[i], iAmax_2h_demo[i], iMmax_demo[i]]/14.)+'$', fontsize=20)
    #if i != 0: plt.text(1.7E1, -2.0E0, r'$'+'%4.2f' % (Chi2[iAmax_1h_demo[i], iAmax_2h_demo[i], iMmax_demo[i]]/14.)+'$', fontsize=20)
#   plt.text(1E1, 1.5E0, r'$\boldsymbol{\chi}^2/{dof}='+'%4.2f' % (Chi2[iAmax_1h_15, iAmax_2h_15, nMhalo-1]/14.)+'$', fontsize=20)
    if i==ndemo/2: plt.ylabel(r'$\boldsymbol{\delta \Sigma}_\mathrm{Mg\,II}/\boldsymbol{\Sigma}_\mathrm{Mg\,II}^\mathrm{model}$', rotation='vertical', fontsize=26)

plt.setp(ax.get_xticklabels(), visible=True)
plt.xticks([0.1, 1, 10],('100 kpc', '1 Mpc', '10 Mpc'), fontsize=22)
plt.xlabel(r'$\mathbf{r}_\mathrm{p}$', fontsize=26)
plt.show()
plt.savefig('Halomodel_demo_bestfit_residuals_sat1.eps')
