
import time
import numpy as np
import halomodel as hm

CosPar = {'Omega_M':0.3, 'Omega_L':0.7, 'Omega_b':0.045, 'Omega_nu':1e-5, 'n_degen_nu':3., 'h':0.7, 'sigma_8':0.8, 'ns':0.96}
z = 0.52

k_min = 1E-6
k_max = 1E4
dlogk = 1.E-2
kk = np.exp(np.arange(np.log(k_min)-2.*dlogk,np.log(k_max)+2.*dlogk,dlogk))

time0=time.time()
#pk2h = hm.ps_2h_gal_dm(kk, 1E13, z, CosPar, doHalo=True)
pk2h = hm.ps_2h_gal_dm(kk, 1E13, z, CosPar, doHalo=False)
pk2h = pk2h.reshape(pk2h.size)
time1=time.time()
print "Elapsed time: ", (time1-time0)/60.

#np.savetxt('pk_2h_no_bias_z0.52.dat', zip(kk, pk2h), fmt='%G  %G')
np.savetxt('linear_pk_2h_no_bias_z0.52.dat', zip(kk, pk2h), fmt='%G  %G')
