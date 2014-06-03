
import time
import numpy as np
import halomodel as hm

CosPar = {'Omega_M':0.3, 'Omega_L':0.7, 'Omega_b':0.045, 'Omega_nu':1e-5, 'n_degen_nu':3., 'h':0.7, 'sigma_8':0.8, 'ns':0.96}
z = 0.52
#data = np.genfromtxt('pk_2h_no_bias_z0.52.dat', dtype=[('k', 'f'), ('pk', 'f')])
data = np.genfromtxt('linear_pk_2h_no_bias_z0.52.dat', dtype=[('k', 'f'), ('pk', 'f')])
pk = data['pk']

R_min = 1E-5 
R_max = 3E2
dlogR = 1.E-2
RR = np.exp(np.arange(np.log(R_min)-2.*dlogR,np.log(R_max)+2.*dlogR,dlogR))

k_min = 1E-6
k_max = 1E4
dlogk = 1.E-2
kk = np.exp(np.arange(np.log(k_min)-2.*dlogk,np.log(k_max)+2.*dlogk,dlogk))
nlogk = kk.size

kR = RR*kk.reshape(kk.size, 1) # kR(k, RR)
WRK = np.sin(kR)/kR
Ifbt = kk.reshape(kk.size, 1)**3/2./np.pi**2*pk.reshape(kk.size, 1)*WRK
#Ifbt = kk**3/2./np.pi**2*pk*WRK

xiR = np.sum((Ifbt[2:,:]+Ifbt[:nlogk-2,:]+4.*Ifbt[1:nlogk-1,:])/6.*dlogk, axis=0)

# np.savetxt('xiR_2h_no_bias_z0.52.dat', zip(RR, xiR), fmt='%G  %G')
np.savetxt('linear_xiR_2h_no_bias_z0.52.dat', zip(RR, xiR), fmt='%G  %G')
# comoving
