
""" 
Halo Model
- Everything in Physical frame, except for p(k) and xi(R) - pk, k, xi, R all comoving
- To-do: write a vectorized Simpson integration routine
- at >30 Mpc not accurate
"""

# TO-DO:
# 2-halo term for Sigma
# Need to tabulate everything in the end

import sys
import numpy as np
from scipy import fftpack
from scipy import integrate 
from scipy import interpolate
from scipy.special import hyp2f1, spence
import cosmology as cosmo
import powerspectra as ps

circular_virial = lambda Mhalo, z, CosPar: np.sqrt(cosmo.GG_MSun*Mhalo/virial_radius(Mhalo, 0.52, CosPar))
delta_sc = lambda z, CosPar: 3./20.*pow(12.*np.pi, 2./3.)*(1.+0.013*np.log10(cosmo.Omega_M_z(z, CosPar)))
concentration = lambda Mhalo, z, Mhalo_star: 9./(1.+z)*(Mhalo/Mhalo_star)**(-0.13) # There are other models Hu & Kravtsov
sigma_M_sqr = lambda Mhalo, z, CosPar: ps.sigma_R_sqr(pow(3.*Mhalo/4./np.pi/cosmo.rho_critical(z, CosPar)/cosmo.Omega_M_z(z,CosPar), 1./3.)*(1.+z), z, CosPar) # Physical to Comoving radius
sigma_j_M_sqr = lambda j, Mhalo, z, CosPar: ps.sigma_j_R_sqr(j, pow(3.*Mhalo/4./np.pi/cosmo.rho_critical(z, CosPar)/cosmo.Omega_M_z(z,CosPar), 1./3.)*(1.+z), z, CosPar) # Physical to Comoving radius
sigma_j_r_M1_M2_sqr = lambda j, r, Mhalo1, Mhalo2, z, CosPar: sigma_j_r_R1_R2_sqr(j, r, pow(3.*Mhalo1/4./np.pi/cosmo.rho_critical(z, CosPar)/cosmo.Omega_M_z(z,CosPar), 1./3.)*(1.+z), pow(3.*Mhalo2/4./np.pi/cosmo.rho_critical(z, CosPar)/cosmo.Omega_M_z(z,CosPar), 1./3.)*(1.+z), z, CosPar) # Physical to Comoving radius
nu_f_nu = lambda nu: 0.129*np.sqrt(nu/np.sqrt(2.))*(1.+pow(nu/np.sqrt(2.), -0.3))*np.exp(-nu/np.sqrt(2.)/2.) # Sheth & Tormen 1999 nu=[delta_sc/D_growth/sigma_M]^2. 0.129 is for the whole integral, need to calculate the normalization again
bias_nu = lambda nu, d_sc: 1.+nu/np.sqrt(2.)/d_sc+0.35*pow(nu/np.sqrt(2.), 1.-0.8)/d_sc-pow(nu/np.sqrt(2.), 0.8)/(pow(nu/np.sqrt(2.), 0.8)+0.35*(1.-0.8)*(1.-0.8/2.))/d_sc*np.sqrt(np.sqrt(2.)) # Tinker+2005
bias_nu_st = lambda nu, d_sc: 1.+(0.73*nu-1.)/d_sc+2.*0.15/d_sc/(1.+pow(0.73*nu, 0.15)) # Sheth & Tormen 1999
f_sigma = lambda sigma_M: 0.186*(1.+pow(sigma_M/2.57, -1.47))*np.exp(-1.19/sigma_M**2) # Tinker+2008, Delta=200
f_Legendre = lambda z, CosPar: cosmo.Omega_M_z(z, CosPar)**0.55

#Mhalo_star: This has to be tabulated for different cosmology parameters. Calculate it on the fly takes too much time. (Compute_Mhalo_star(Cosmo))
M_star = 5.19E12 # MSun not h^-1 MSun

# Even though it's vectorized, it requires a large amount of memory to have a 3-D array (1E4, 1E4, 1E4)
def NFW_ukm_integrand(logR, k, Mhalo, z, CosPar):
    """
    NFW_ukm_integrand(logR, kk, Mhalo, z, CosPar)
    """

    R_vir = virial_radius(Mhalo, z, CosPar)
    RR = R_vir*(np.exp(logR)).reshape(logR.shape[0], logR.size/logR.shape[0])

    rhoR = NFW_profile(RR, Mhalo, z, CosPar) # rhoR(RR,Mhalo)
    RR_tmp = np.ones(Mhalo.size)*RR
    kR = RR_tmp*k.reshape(k.size, 1, 1) # kR(k, RR, Mhalo)

    WRK = np.sin(kR)/kR # (k, RR, Mhalo)
    return 1E18*4.*np.pi*RR_tmp**3*WRK*rhoR

def NFW_ukm(k, Mhalo, z, CosPar):
    """
    NFW_ukm(k, Mhalo, z, CosPar): output ukm[k, Mhalo]
    """
    k = np.array(k)
    Mhalo = np.array(Mhalo)

    R_min = 1E-4 # in R_vir in the integrand
    R_max = 1.
    dlogR = 1E-3
    logR = np.arange(np.log(R_min)-2.*dlogR, np.log(R_max)+2.*dlogR, dlogR)
    nlogR = logR.size
    Integrand = NFW_ukm_integrand(logR, k, Mhalo, z, CosPar) # (k, RR, Mhalo)

    return np.sum((Integrand[:,2:,:]+Integrand[:,:nlogR-2,:]+4.*Integrand[:,1:nlogR-1,:])/6.*dlogR, axis=1)/Mhalo

def ps_2h_gal_dm_integrand(logM, k, z, CosPar):
    """
    ps_2h_gal_dm_integrand(logM, k, z, CosPar):
    """
    MM = np.exp(logM)
    dn_dlogM = halo_mass_function(MM, z, CosPar)
    bM = bias(MM, z, CosPar)

    # This is because I don't have 1 TB RAM
    ukm = np.zeros((k.size, logM.size))

    progressbar_width = 80
    progressbar_interval = logM.size/progressbar_width+1

    # setup progress bar
    sys.stdout.write("[%s]" % (" " * progressbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (progressbar_width+1)) # return to start of line, after '['

    # Note ukm is physical, but k is comoving
    for i in np.arange(logM.size): 
        ukm[:,i] = (NFW_ukm(k*(1.+z), MM[i], z, CosPar)).reshape(k.size)
        if (i%progressbar_interval==0):
            sys.stdout.write("-")
            sys.stdout.flush()
    sys.stdout.write("\n")

    return dn_dlogM*bM*ukm*MM/cosmo.rho_critical(z, CosPar)/cosmo.Omega_M_z(z, CosPar)

# No bias included
def ps_2h_gal_dm(k, Mhalo, z, CosPar, doHalo=False):
    """
    ps_2h_gal_dm(k, Mhalo, z, CosPar): return pk(k, Mhalo)
    """
    k = np.array(k)
    bias_normalization = 1.08037 # The mean bias between 1E3 and 1E17 solar mass
    #bM = bias(Mhalo, z, CosPar)
    #print bM
    plk = ps.ps_linear(k, z, CosPar)

    if doHalo:
        # This needs to be changed for spatial exclusion and normalization -- so we keep Mhalo in the arguments
        M_min = 1E3
        M_max = 1E17
        dlogM = 1E-2
        logM = np.arange(np.log(M_min)-2.*dlogM, np.log(M_max)+2.*dlogM, dlogM)
        nlogM = logM.size
        Integrand = ps_2h_gal_dm_integrand(logM, k, z, CosPar)
        return (plk*np.sum((Integrand[:,2:]+Integrand[:,:nlogM-2]+4.*Integrand[:,1:nlogM-1])/6.*dlogM, axis=1)).reshape(k.size, 1)/bias_normalization
    else:
        return plk.reshape(k.size, 1)

def xi_from_pk_fbt(R, Pk_func, UseTable=False):
    """
    xi_from_pk_fbt(R, Pk_func): Inverse Fourier Bessel Transform
    """
    k_min = 1E-6
    k_max = 1E4
    dlogk = 1.E-2
    kk = np.exp(np.arange(np.log(k_min)-2.*dlogk,np.log(k_max)+2.*dlogk,dlogk))
    nlogk = kk.size

    pk = Pk_func(kk) # pk(kk, Mhalo)
    kR = kk.reshape(kk.size, 1, 1)*R.reshape(1, R.shape[0], R.shape[0]/R.size) # kR(k, RR, Mhalo)
    WRK = np.sin(kR)/kR
    Ifbt = kk.reshape(kk.size, 1, 1)**3/2./np.pi**2*pk.reshape(pk.shape[0], 1, pk.shape[1])*WRK

    return np.sum((Ifbt[2:,:,:]+Ifbt[:nlogk-2,:,:]+4.*Ifbt[1:nlogk-1,:,:])/6.*dlogk, axis=0)

# No bias included: xiR = b(Mhalo)*xi_2h_gal_dm
# R can be in shape of (R, Mhalo) or (R)
def xi_2h_gal_dm(R, Mhalo, z, CosPar, doHalo=False):
    """
    xi_2h_gal_dm(R, Mhalo, z, CosPar):
    """
    Mhalo = np.array(Mhalo)
    Mhalo = Mhalo.reshape(Mhalo.size)
    R = np.ones(Mhalo.size)*R.reshape(R.shape[0], R.size/R.shape[0])
    xiR = np.zeros((R.size, Mhalo.size))
    # This is because I don't have 1TB memory
    for i in np.arange(Mhalo.size): 
        Pk_func = lambda k: ps_2h_gal_dm(k, Mhalo[i], z, CosPar, doHalo=doHalo) 
        xiR[:,i] = (xi_from_pk_fbt(R[:,i], Pk_func)).reshape(R.shape[0])
    # bM = bias(Mhalo, z, CosPar)
    return xiR

# Mhalo_star in when Sigma = delta_sc
def Mhalo_star(z, CosPar):
    M_min = 1E8
    M_max = 1E17
    dlogM = 1E-2
    logM = np.arange(np.log(M_min), np.log(M_max)+2.*dlogM, dlogM)
    MM = np.exp(logM)
    sigma_M = np.sqrt(sigma_M_sqr(MM, 0, CosPar)) # at z=0
    dsc = delta_sc(z, CosPar)

    f = interpolate.interp1d(sigma_M[::-1], MM[::-1])
    return f(dsc)

# b(M)
def bias(Mhalo0, z, CosPar):
    Mhalo = np.array(Mhalo0)
    print "IF YOU SEE ME TOO MANY TIMES, YOU SHOULD VECTORIZE YOUR CODE. MAYBE YOU SHOULD VECTORIZE ME!"
    M_min = 1E3
    M_max = 1E17
    #if min(Mhalo)<M_min: raise ValueError("halo mass too small")
    #if max(Mhalo)>M_max: raise ValueError("halo mass too large")

    dlogM = 1E-2
    logM = np.arange(np.log(M_min)-2.*dlogM, np.log(M_max)+2.*dlogM, dlogM)
    MM = np.exp(logM)
    sigma_M2 = sigma_M_sqr(MM, 0, CosPar) # at z=0
    dsc = delta_sc(z, CosPar)
    nuM = dsc**2/sigma_M2/(cosmo.D_growth(z,CosPar)/cosmo.D_growth(0,CosPar))**2

    bM = bias_nu(nuM, dsc)

    f = interpolate.interp1d(MM, bM)
    return f(Mhalo)

# dN/dlogM
def halo_mass_function(Mhalo, z, CosPar):
    print "IF YOU SEE ME TOO MANY TIMES, YOU SHOULD VECTORIZE YOUR CODE. MAYBE YOU SHOULD VECTORIZE ME!"
    M_min = 1E3
    M_max = 1E17

    dlogM = 1E-2
    logM = np.arange(np.log(M_min)-2.*dlogM, np.log(M_max)+2.*dlogM, dlogM)
    nlogM = logM.size
    MM = np.exp(logM)
    sigma_M2 = sigma_M_sqr(MM, 0, CosPar) # at z=0
    dsc = delta_sc(z, CosPar)
    nuM = dsc**2/sigma_M2/(cosmo.D_growth(z,CosPar)/cosmo.D_growth(0,CosPar))**2
    #print min(nuM), max(nuM)
    dlognuM = np.zeros(nlogM) 
    dlognuM[1:] = np.log(nuM[1:]/nuM[:nlogM-1])
    dlognuM[0] = dlognuM[1]

    fM = nu_f_nu(nuM)
    nmz = fM*cosmo.rho_critical(z, CosPar)*cosmo.Omega_M_z(z, CosPar)/MM*dlognuM/dlogM
    Inte = fM*dlognuM
    normalization = np.sum(Inte[2:]+Inte[:nlogM-2]+4.*Inte[1:nlogM-1])/6.
    #print normalization

    f = interpolate.interp1d(MM, nmz)
    return f(Mhalo)/normalization

def Delta_virial(z, CosPar):
    """
    Delta_virial(z, CosPar): Bryan & Norman 1998. See also Weinberg & Kamionkowski 2003
    """
    #x = cosmo.Omega_M_z(z, CosPar)*(1.+z)**3/(cosmo.Omega_M_z(z,CosPar)*(1.+z)**3+cosmo.Omega_L_z(z,CosPar))
    x = cosmo.Omega_M_z(z, CosPar)-1.
    return (18*np.pi*np.pi+82.*x-39.*x*x)/(1.+x)

# Is this comoving or physical?: Physical
def virial_radius(Mhalo, z, CosPar):
    """
    virial_radius(Mhalo, z, CosPar): physical virial radius, multiply it by 1.+z to get comoving radius
    """
    factor = 3./4./np.pi/cosmo.rho_critical(z, CosPar)/cosmo.Omega_M_z(z,CosPar)/Delta_virial(z, CosPar)
    return (factor*Mhalo)**(1./3.)

# Bullock 2001 approximation
virial_radius_jb01 = lambda Mhalo, z, CosPar: pow(Mhalo/1E11*CosPar['h']*200./CosPar['Omega_M']/Delta_virial(z, CosPar), 1./3.)*75./(1.+z)/CosPar['h']

def rho_s(c, z, CosPar, alpha=1.):
    """
    rho_s(c, z, CosPar, alpha=1.)
    """
    if (alpha<3. and alpha>0.):
        if (alpha==1.):
            factor=(np.log(1.+c)-c/(1.+c))
        else:
            factor = c**(3.-alpha)/(3.-alpha)*hyp2f1(3.-alpha, 3.-alpha, 4.-alpha, -c)
    else:
        raise ValueError("alpha has to be in the set (0,3)")

    return cosmo.rho_critical(z,CosPar)*cosmo.Omega_M_z(z, CosPar)*Delta_virial(z, CosPar)*c**3/3./factor

# NFW profile, allowing concentration to be fcorr*c and the slope to be alpha
# r in ascending order
# rho(r, Mhalo)
def NFW_profile(R, Mhalo, z, CosPar, fcorr=1., alpha=1.):
    """
    NFW_profile(R, Mhalo, z, CosPar, fcorr=1., alpha=1.)
    return rho(R), R in Mpc, rho in MSun pc^-2
    """
    R = np.array(R) # r could be r(r) or r(r, Mhalo)
    R = R.reshape(R.shape[0], R.size/R.shape[0])
    Mhalo = np.array(Mhalo)
    c = fcorr*concentration(Mhalo, z, M_star)
    R_vir = virial_radius(Mhalo, z, CosPar)
    rhos = rho_s(c,z,CosPar,alpha)

    x = c*R/R_vir # cr/rvir(R, Mhalo)
    rsqrrho = pow(x, -alpha)*pow(1.0+x, alpha-3.0)

    return rhos*rsqrrho/1E18 # 1E18 Mpc -> pc

# y can be y(ny,1) or y(ny, nMhalo)
def NFW_project_profile(y, Mhalo, z, CosPar, fcorr=1., alpha=1.):
    """
    NFW_project_profile(y0, Mhalo, z, CosPar, fcorr=1., alpha=1.)
    """
    # Need to do some argumente/keyword checking
    R_min = 1E-5
    R_max = 2E3
    if min(y)<R_min: raise ValueError("y too small")
    if max(y)>R_max: raise ValueError("y too large")

    y = np.array(y)
    y = y.reshape(y.shape[0], y.size/y.shape[0]) # Convert it to y(ny,1) or y(ny,nMhalo)
    ny = y.shape[0]
    Mhalo = np.array(Mhalo)
    nMhalo = Mhalo.size
    yy = np.ones(nMhalo)*y.reshape(ny,1)

    # Set up interpolation profiles
    dlogR = 1E-2
    R = np.exp(np.arange(np.log(R_min)-2.*dlogR, np.log(R_max)+2.*dlogR, dlogR))
    nR = R.shape[0]
    R = np.ones(nMhalo)*R.reshape(nR, 1) # R(nR, nMhalo)
    rhoR = NFW_profile(R, Mhalo, z, CosPar, fcorr, alpha) # rhoR(nR, nMhalo)

    Rout = 6.
    Rvir = virial_radius(Mhalo, z, CosPar).reshape(nMhalo) # Rvir(nMhalo)

    Sigma_y = np.zeros((ny, nMhalo)) # Initialize output: Sigma_y(ny, nMhalo)

    # Very difficult to vectorize without approximation
    # Let's settle for what we have now
    dlogs = 1E-3 # 10% better than 1E-2
    for i in np.arange(nMhalo):
        f = interpolate.interp1d(R[:,i], rhoR[:,i])
        Sigma_integrand = lambda R, s: 2.*f(R)*R**2/np.sqrt(R*R-s*s) # dlogR
        s_max = max(Rvir[i]*Rout, max(yy[:,i])*1.2)
        s_min = yy[:,i]
        for j in np.arange(ny):
            ss = np.exp(np.arange(np.log(s_min[j])+dlogs,np.log(s_max)+dlogs,dlogs))
            nlogs = ss.size
            Integrand = Sigma_integrand(ss, s_min[j])
            Sigma_y[j, i] = np.sum((Integrand[2:]+Integrand[:nlogs-2]+4.*Integrand[1:nlogs-1])/6.*dlogs)*1E6

    return Sigma_y

# This piece of code is not finished yet
def NFW_average_project_density(y0, Mhalo, z, CosPar, fcorr=1., alpha=1.):
    """
    NFW_average_project_density(y0, Mhalo, z, CosPar, fcorr=1., alpha=1.)
    """
    y = np.array(y0)
    c = fcorr*concentration(Mhalo, z, M_star)
    rvir = virial_radius(Mhalo, z, CosPar)
    ymin = rvir/c*1E-4
    ymax = max(y)
    dlogy = 1E-2
    ytmp = np.arange(np.log(ymin)-2.*dlogy, np.log(ymax)+2.*dlogy, dlogy)
    ytmp = np.exp(ytmp)

    if (ytmp.size <= 1): raise ValueError("The radius array is unrealistic.")

    #print "Integration Steps: ", ytmp.size
    #print "Computing projected density:"

    projected_density = NFW_project_profile_novector(ytmp, Mhalo, z, CosPar, fcorr, alpha)

    #print "Done computing projected density Done"
    #print "Computing average density:"

    total_mass = np.zeros(ytmp.size)
    total_mass[0] = projected_density[0]*np.pi*ymin*ymin

    #trapezoidal integration
    for i in np.arange(ytmp.size-1)+1:
        total_mass[i] = total_mass[i-1]+(projected_density[i]*ytmp[i]+projected_density[i-1]*ytmp[i-1])*(ytmp[i]-ytmp[i-1])*np.pi
#       average_density[i] = total_mass[i]/np.pi/ytmp[i]/ytmp[i]

    # Needs testing
    average_density = total_mass/np.pi/ytmp**2

    #intepolation
    f = interpolate.interp1d(ytmp, average_density)
    return f(y)

def NFW_profile_novector(R, Mhalo, z, CosPar, fcorr=1., alpha=1.):
    """
    NFW_profile(R, Mhalo, z, CosPar, fcorr=1., alpha=1.)
    return rho(R), R in Mpc, rho in MSun pc^-2
    """
    c = fcorr*concentration(Mhalo, z, M_star)
    R_vir = virial_radius(Mhalo, z, CosPar)
    rhos = rho_s(c,z,CosPar,alpha)

    x = c*R/R_vir # cr/rvir(R, Mhalo)
    rsqrrho = pow(x, -alpha)*pow(1.0+x, alpha-3.0)

    return rhos*rsqrrho/1E18 # 1E18 Mpc -> pc


# Projected NFW profile, Abel's tranform. *MUST* be vectorized!
def NFW_project_integrand_novector(r, s, Mhalo, z, CosPar, fcorr=1., alpha=1.):
    return 2.*NFW_profile_novector(r, Mhalo, z, CosPar, fcorr, alpha)*r/np.sqrt(r*r-s*s)*1E6
    
# Can/Should use interpolation to speed up (see average_project_density below) *MUST* be vectorized!
def NFW_project_profile_novector(y0, Mhalo, z, CosPar, fcorr=1., alpha=1.):
    """
    NFW_project_profile_novector(y0, Mhalo, z, CosPar, fcorr=1., alpha=1.)
    """
    rout = 5.
    y = np.array(y0)
    rvir = virial_radius(Mhalo, z, CosPar)
    ymax = max(max(y)*1.1, rvir*rout)

    # For each y, call NFW_profile once, can we just call NFW_profile once and be done with it?
    NFW_func = np.vectorize(lambda y, Mhalo, z, CosPar, fcorr, alpha:
        integrate.quad(NFW_project_integrand_novector, y*1.001, ymax, limit=1000,
        args=(y, Mhalo, z, CosPar, fcorr, alpha)))
    NFW_proj, err = NFW_func(y, Mhalo, z, CosPar, fcorr, alpha)

    return NFW_proj

# Bottleneck, NFW_project_profile *MUST* be vectorized
def NFW_average_project_density_novector(y0, Mhalo, z, CosPar, fcorr=1., alpha=1.):
    """
    NFW_average_project_density_novector(y0, Mhalo, z, CosPar, fcorr=1., alpha=1.)
    """
    y = np.array(y0)
    c = fcorr*concentration(Mhalo, z, M_star)
    rvir = virial_radius(Mhalo, z, CosPar)
    ymin = rvir/c*1E-4
    ymax = max(y)
    dlogy = 1E-2
    ytmp = np.arange(np.log(ymin)-2.*dlogy, np.log(ymax)+2.*dlogy, dlogy)
    ytmp = np.exp(ytmp)

    if (ytmp.size <= 1): raise ValueError("The radius array is unrealistic.")

    #print "Integration Steps: ", ytmp.size
    #print "Computing projected density:"

    projected_density = NFW_project_profile_novector(ytmp, Mhalo, z, CosPar, fcorr, alpha)

    #print "Done computing projected density Done"
    #print "Computing average density:"

    total_mass = np.zeros(ytmp.size)
    total_mass[0] = projected_density[0]*np.pi*ymin*ymin

    #trapezoidal integration
    for i in np.arange(ytmp.size-1)+1:
        total_mass[i] = total_mass[i-1]+(projected_density[i]*ytmp[i]+projected_density[i-1]*ytmp[i-1])*(ytmp[i]-ytmp[i-1])*np.pi
#       average_density[i] = total_mass[i]/np.pi/ytmp[i]/ytmp[i]

    # Needs testing
    average_density = total_mass/np.pi/ytmp**2

    #intepolation
    f = interpolate.interp1d(ytmp, average_density)
    return f(y)

# One dimensional virial motion?
def NFW_approx_sigma_virial(Mhalo, z, CosPar, fcorr=1., alpha=1.):
    fsigma = 0.9
    corr = (cosmo.Hubble_z(z, CosPar)**2*Delta_virial(z,CosPar))
    return 476.*fsigma*pow(Mhalo/1E15, 1/3.)*pow(corr, 1./6.)

def NFW_approx_sigma_halo(Mhalo, z, CosPar, fcorr=1., alpha=1.):
    sigma_fit = 400.
    R_fit = 50.
    #R_scale = virial_radius(Mhalo, z, CosPar)
    R_scale = pow(3.*Mhalo/4./np.pi/cosmo.rho_critical(z, CosPar)/cosmo.Omega_M_z(z,CosPar), 1./3.)*(1.+z)
    #print R_scale
    eta = 0.85
    return sigma_fit/(1.+pow(R_scale/R_fit, eta))

def NFW_sigma_halo(Mhalo, z, CosPar, fcorr=1., alpha=1.):
    return np.sqrt(sigma_j_M_sqr(-1, Mhalo, z, CosPar))*CosPar['h']*100*f_Legendre(0., CosPar)*np.sqrt(1.-sigma_j_M_sqr(0, Mhalo, z, CosPar)**2/sigma_j_M_sqr(1,Mhalo,z,CosPar)/sigma_j_M_sqr(-1,Mhalo,z,CosPar))

# velocity anisotropy: beta = 0.5 (radial velocity dispersion)
def NFW_sigma(R, Mhalo, z, CosPar, fcorr=1., alpha=1.):
    """
    NFW_sigma(R, Mhalo, z, CosPar, fcorr=1., alpha=1.)
    return rho(R), R in Mpc, rho in MSun pc^-2
    """
    R = np.array(R) # r could be r(r) or r(r, Mhalo)
    R = R.reshape(R.shape[0], R.size/R.shape[0])
    Mhalo = np.array(Mhalo)
    c = fcorr*concentration(Mhalo, z, M_star)
    R_vir = virial_radius(Mhalo, z, CosPar)
    rhos = rho_s(c,z,CosPar,alpha)

    x = c*R/R_vir # cr/rvir(R, Mhalo)
    
    gc = 1./(np.log(1.+c)-c/(1.+c))
    return circular_virial(Mhalo,z,CosPar)*(1.+x)*np.sqrt(0.5*x*c*gc*(np.pi**2-np.log(x)-1./x-1./(1.+x)**2-6./(1.+x)+(1.+1./x/x-4./x-2./(1.+x))*np.log(1.+x)+3.*pow(np.log(1.+x),2)+6.*spence(1.+x)))
    #return circular_virial(Mhalo,z,CosPar)*(1.+x)*np.sqrt(c*gc*(-np.pi**2/3.+1./2./(1.+x)**2+2./(1.+x)+np.log(1.+x)/x+np.log(1.+x)/(1.+x)-pow(np.log(1.+x),2)-2.*spence(1.+x)))
    #return circular_virial(Mhalo,z,CosPar)*(1.+x)*np.sqrt(c*gc/x*(np.pi**2/6.-1./2./(1.+x)**2-1./(1.+x)-np.log(1.+x)/(1.+x)+pow(np.log(1.+x),2)/2.+spence(1.+x)))

def NFW_project_sigma(y, Mhalo, z, CosPar, fcorr=1., alpha=1.):
    """
    NFW_project_sigma(y0, Mhalo, z, CosPar, fcorr=1., alpha=1.)
    """
    # Need to do some argumente/keyword checking
    bbeta = 0.0
    R_min = 1E-5
    R_max = 2E3
    if min(y)<R_min: raise ValueError("y too small")
    if max(y)>R_max: raise ValueError("y too large")

    y = np.array(y)
    y = y.reshape(y.shape[0], y.size/y.shape[0]) # Convert it to y(ny,1) or y(ny,nMhalo)
    ny = y.shape[0]
    Mhalo = np.array(Mhalo)
    nMhalo = Mhalo.size
    yy = np.ones(nMhalo)*y.reshape(ny,1)

    # Set up interpolation profiles
    dlogR = 1E-2
    R = np.exp(np.arange(np.log(R_min)-2.*dlogR, np.log(R_max)+2.*dlogR, dlogR))
    nR = R.shape[0]
    R = np.ones(nMhalo)*R.reshape(nR, 1) # R(nR, nMhalo)
    rhoR = NFW_profile(R, Mhalo, z, CosPar, fcorr, alpha)*NFW_sigma(R, Mhalo, z, CosPar, fcorr, alpha)**2 # sigma^2*rhoR(nR, nMhalo)

    Rout = 5.
    Rvir = virial_radius(Mhalo, z, CosPar).reshape(nMhalo) # Rvir(nMhalo)

    Sigma_y = np.zeros((ny, nMhalo)) # Initialize output: Sigma_y(ny, nMhalo)

    # Very difficult to vectorize without approximation
    # Let's settle for what we have now
    dlogs = 1E-3 # 10% better than 1E-2
    for i in np.arange(nMhalo):
        f = interpolate.interp1d(R[:,i], rhoR[:,i])
        Sigma_integrand = lambda R, s: (1.-bbeta*s*s/R/R)*2.*f(R)*R**2/np.sqrt(R*R-s*s) # dlogR
        s_max = max(Rvir[i]*Rout, max(yy[:,i])*1.1)
        s_min = yy[:,i]
        for j in np.arange(ny):
            ss = np.exp(np.arange(np.log(s_min[j])+dlogs,np.log(s_max)+dlogs,dlogs))
            nlogs = ss.size
            Integrand = Sigma_integrand(ss, s_min[j])
            Sigma_y[j, i] = np.sum((Integrand[2:]+Integrand[:nlogs-2]+4.*Integrand[1:nlogs-1])/6.*dlogs)*1E6

    Norm_Sigma_y = NFW_project_profile(y, Mhalo, z, CosPar, fcorr, alpha)

    return np.sqrt(Sigma_y/Norm_Sigma_y)

# Premier for 2-halo term velocity correlation
# Only vectorize in R2
#def sigma_j_R1_R2_integrand_unnorm(j, logk, R1, R2, z, CosPar):
#    kk = np.exp(logk)
#    Tk = kk**(3+2.*j)*ps.EH_TF_mpc(kk, z, CosPar) #/2./np.pi**2*(3E5/70.)**(3.95)*(1.95E-5*pow(0.27, -0.785-0.05*np.log(0.27))*np.exp(-1.14))**2
#    kR1 = kk*(np.ones(R2.size)*R1.reshape(R1.size, 1)).reshape(R1.size, R2.size, 1) #Be careful with the broadcasting rules 
#    kR2 = kk*(np.ones(R1.size)*R2.reshape(1, R2.size)).reshape(R1.size, R2.size, 1) #Be careful with the broadcasting rules 
#    # Top-hat window function
#    WRK1 = 3./kR1**3*(np.sin(kR1)-kR1*np.cos(kR1))
#    WRK2 = 3./kR2**3*(np.sin(kR2)-kR2*np.cos(kR2))
#    return Tk*WRK1*WRK2

#def sigma_j_r_R1_R2_sqr_unnorm(j, r, R1, R2, z, CosPar):
#    """
#    sigma_j_r_R1_R2_sqr_unnorm(j, r, R1, R2, z, CosPar):
#    """
#    rr = np.array(r)
#    RR1 = np.array(R1)
#    RR2 = np.array(R2)
#    k_min = 1E-6
#    k_max = 1E3
#    dlogk = 1.E-2
#    logk = np.arange(np.log(k_min),np.log(k_max)+2.*dlogk,dlogk)
#    nlogk = logk.size
#    # [nr, nR1, nR2, nk]
#    kr = np.exp(logk)*((rr.reshape(rr.size, 1)*np.ones(RR1.size)).reshape(rr.size, RR1.size, 1)*np.ones(RR2.size)).reshape(rr.size, RR1.size, RR2.size, 1)
#    KRK = (np.sin(kr)/kr)-(2./kr**3)*(np.sin(kr)-kr*np.cos(kr))
#    Integrand = (sigma_j_R1_R2_integrand_unnorm(j, logk, RR1, RR2,z, CosPar)).reshape(1, RR1.size, RR2.size, nlogk)*KRK
#
    # Compute the integral (Maybe a vectorized trapezoidal integral would be faster?)
    #print "Integrating..."
    # return np.sum((Integrand[:,1:]+Integrand[:,:nlogk-1])/2.*dlogk, axis=1)
#   return np.sum((Integrand[:,2:]+Integrand[:,:nlogk-2]+4.*Integrand[:,1:nlogk-1])/6.*dlogk, axis=1)

def sigma_j_R1_R2_integrand_unnorm(j, logk, R1, R2, z, CosPar):
    kk = np.exp(logk)
    Tk = kk**(3+2.*j)*ps.EH_TF_mpc(kk, z, CosPar) #/2./np.pi**2*(3E5/70.)**(3.95)*(1.95E-5*pow(0.27, -0.785-0.05*np.log(0.27))*np.exp(-1.14))**2
    kR1 = kk*R1 #Be careful with the broadcasting rules 
    kR2 = kk*R2.reshape(R2.size, 1) #Be careful with the broadcasting rules 
    # Top-hat window function
    WRK1 = (3./kR1**3*(np.sin(kR1)-kR1*np.cos(kR1))).reshape(1, kk.size)
    WRK2 = 3./kR2**3*(np.sin(kR2)-kR2*np.cos(kR2))
    return Tk*WRK1*WRK2

def sigma_j_r_R1_R2_sqr_unnorm(j, r, R1, R2, z, CosPar, radial=False):
    """
    sigma_j_r_R1_R2_sqr_unnorm(j, r, R1, R2, z, CosPar):
    """
    rr = np.array(r)
    RR1 = np.array(R1)
    if RR1.size>1: raise ValueError("r and R1 should be scalar")
    RR2 = np.array(R2)
    k_min = 1E-6
    k_max = 1E3
    dlogk = 1.E-2
    logk = np.arange(np.log(k_min),np.log(k_max)+2.*dlogk,dlogk)
    nlogk = logk.size
    # [nr, nR1, nR2, nk]
    kr = (np.exp(logk)).reshape(1, nlogk)*rr.reshape(rr.size, 1)
    if radial: 
        KRK = (np.sin(kr)/kr)-(2./kr**3)*(np.sin(kr)-kr*np.cos(kr))
    else:
        KRK = (np.sin(kr)/kr)
    #   KRK = (1./kr**3)*(np.sin(kr)-kr*np.cos(kr))
    Integrand = (sigma_j_R1_R2_integrand_unnorm(j, logk, R1, RR2, z, CosPar)).reshape(1, RR2.size, nlogk)*KRK.reshape(rr.size, 1, nlogk)

   # Compute the integral (Maybe a vectorized trapezoidal integral would be faster?)
   #print "Integrating..."
   # return np.sum((Integrand[:,1:]+Integrand[:,:nlogk-1])/2.*dlogk, axis=1)
    return np.sum((Integrand[:,:,2:]+Integrand[:,:,:nlogk-2]+4.*Integrand[:,:,1:nlogk-1])/6.*dlogk, axis=2)

def sigma_j_r_R1_R2_sqr(j, r, R1, R2, z, CosPar, radial=False):
    srR12_sqr_unnormalized = sigma_j_r_R1_R2_sqr_unnorm(j, r, R1, R2, z, CosPar, radial=radial)
    s8_sqr = ps.sigma_R_sqr_unnorm(8./CosPar['h'], 0., CosPar)
    return srR12_sqr_unnormalized*CosPar['sigma_8']**2/s8_sqr

# We only need to calculate it once ...
def NFW_2h_sigma_nocorr(R0, Mhalo0, z, CosPar, mu=1., fcorr=1., alpha=1.):
    Mhalo = np.array(Mhalo0)
    nMhalo = Mhalo.size
    Mhalo = Mhalo.reshape(nMhalo)
    #if nMhalo>1: raise ValueError("Mhalo must be a scalar")
    R = np.array(R0)
    nR = R.size
    R = R.reshape(nR)
    RR = np.ones(nMhalo)*R.reshape(nR,1)

    sigma_halo = NFW_approx_sigma_halo(Mhalo, z, CosPar)
    bMhalo = bias(Mhalo, z, CosPar)

    # get \xi(R) no bias
    xiR = xi_2h_gal_dm(R, 1E13, z, CosPar)

    # Integration over all masses
    M_min = 1E3
    M_max = 1E17
    dlogM = 1E-2
    logM = np.arange(np.log(M_min)-2.*dlogM, np.log(M_max)+2.*dlogM, dlogM)
    MM = np.exp(logM)
    dn_dlogM = halo_mass_function(MM, z, CosPar)
    bM = bias(MM, z, CosPar)
    nlogM = logM.size
    sigma_halo_MM = NFW_approx_sigma_halo(MM, z, CosPar)
    sigma_virial_MM = NFW_approx_sigma_virial(MM, z, CosPar)*np.sqrt(3.) # Use the approximation to speed things up

    Integrand = (dn_dlogM*bM*MM/cosmo.rho_critical(z, CosPar)/cosmo.Omega_M_z(z, CosPar)*(sigma_halo_MM**2+(mu*sigma_virial_MM)**2)).reshape(1,1,nlogM)*(1.+bMhalo.reshape(1, nMhalo, 1)*bM.reshape(1,1,nlogM)*xiR.reshape(nR, nMhalo, 1))
    sigma2_average = np.sum((Integrand[:,:,2:]+Integrand[:,:,:nlogM-2]+4.*Integrand[:,:,1:nlogM-1])/6.*dlogM, axis=2)#.reshape(nR, nMhalo)

    return np.sqrt((sigma_halo**2).reshape(1, nMhalo)+sigma2_average/(1.+bMhalo.reshape(1,nMhalo)*xiR))


def NFW_2h_sigma_allmass(R0, Mhalo0, z, CosPar, mu=1., fcorr=1., alpha=1.):

    Mhalo = np.array(Mhalo0)
    nMhalo = Mhalo.size
    Mhalo = Mhalo.reshape(nMhalo)
    #if nMhalo>1: raise ValueError("Mhalo must be a scalar")
    R = np.array(R0)
    nR = R.size
    R = R.reshape(nR)
    RR = np.ones(nMhalo)*R.reshape(nR,1)

    sigma_halo = NFW_approx_sigma_halo(Mhalo, z, CosPar)
    bMhalo = bias(Mhalo, z, CosPar)

    # get \xi(R) no bias
    xiR = xi_2h_gal_dm(R, 1E13, z, CosPar)

    # This is impossible
    # Integration over all masses
    M_min = 1E3
    M_max = 1E17
    dlogM = 1E-2
    logM = np.arange(np.log(M_min)-2.*dlogM, np.log(M_max)+2.*dlogM, dlogM)
    MM = np.exp(logM)
    dn_dlogM = halo_mass_function(MM, z, CosPar)
    bM = bias(MM, z, CosPar)
    nlogM = logM.size
    sigma_halo_MM = NFW_approx_sigma_halo(MM, z, CosPar)
    sigma_virial_MM = NFW_approx_sigma_virial(MM, z, CosPar)*np.sqrt(3.) # Use the approximation to speed things up

    sigma_correlation_sqr = np.zeros(nR*nMhalo*nlogM).reshape(nR, nMhalo, nlogM)
    sigma_Mhalo_corr = np.sqrt(1.-sigma_j_M_sqr(0, Mhalo, z, CosPar)**2/sigma_j_M_sqr(1,Mhalo,z,CosPar)/sigma_j_M_sqr(-1,Mhalo,z,CosPar))
    sigma_MM_corr = np.sqrt(1.-sigma_j_M_sqr(0, MM, z, CosPar)**2/sigma_j_M_sqr(1,MM,z,CosPar)/sigma_j_M_sqr(-1,MM,z,CosPar))
    prefix = (100.*CosPar['h']*f_Legendre(0,CosPar))**2
    for j in np.arange(nMhalo):
        sigma_correlation_sqr[:,j,:] = sigma_j_r_M1_M2_sqr(-1, R, Mhalo[j], MM, z, CosPar)*prefix*sigma_Mhalo_corr[j]*sigma_MM_corr
        #for i in np.arange(nR):
        #    sigma_correlation_sqr[i,j,:] = sigma_j_r_M1_M2_sqr(-1, R[i], Mhalo[j], MM, z, CosPar)*prefix*sigma_Mhalo_corr[j]*sigma_MM_corr

    # print np.median(np.sqrt(sigma_correlation_sqr))
    # Integrand = (dn_dlogM*bM*MM/cosmo.rho_critical(z, CosPar)/cosmo.Omega_M_z(z, CosPar)*(sigma_halo_MM**2+(mu*sigma_virial_MM)**2)).reshape(1,1,nlogM)*(1.+bMhalo.reshape(1, nMhalo, 1)*bM.reshape(1,1,nlogM)*xiR.reshape(nR, nMhalo, 1))
    Integrand = (dn_dlogM*bM*MM/cosmo.rho_critical(z, CosPar)/cosmo.Omega_M_z(z, CosPar)*((sigma_halo_MM**2+(mu*sigma_virial_MM)**2).reshape(1,1,nlogM)-2*sigma_correlation_sqr))*(1.+bMhalo.reshape(1, nMhalo, 1)*bM.reshape(1,1,nlogM)*xiR.reshape(nR, nMhalo, 1))
    sigma2_average = np.sum((Integrand[:,:,2:]+Integrand[:,:,:nlogM-2]+4.*Integrand[:,:,1:nlogM-1])/6.*dlogM, axis=2)#.reshape(nR, nMhalo)

    return np.sqrt((sigma_halo**2).reshape(1, nMhalo)+sigma2_average/(1.+bMhalo.reshape(1,nMhalo)*xiR))

def NFW_2h_sigma(R0, Mhalo0, z, CosPar, mu=1., fcorr=1., alpha=1.):
    """
    NFW_2h_sigma(R0, Mhalo0, z, CosPar, fcorr=1., alpha=1.): Use only 1E13 as a typical halo mass
    """

    Mhalo = np.array(Mhalo0)
    nMhalo = Mhalo.size
    Mhalo = Mhalo.reshape(nMhalo)
    #if nMhalo>1: raise ValueError("Mhalo must be a scalar")
    R = np.array(R0)
    nR = R.size
    R = R.reshape(nR)
    RR = np.ones(nMhalo)*R.reshape(nR,1)

    # sigma_halo = NFW_approx_sigma_halo(Mhalo, z, CosPar)
    sigma_halo = NFW_sigma_halo(Mhalo, z, CosPar)
    bMhalo = bias(Mhalo, z, CosPar)

    # get \xi(R) no bias
    # xiR = xi_2h_gal_dm(R, 1E13, z, CosPar)

    # This is impossible
    # Integration over all masses
    # Choose one mass (1E13) for approximation
    MM = np.array([1E12])
    nlogM = MM.size
    # sigma_halo_MM = NFW_approx_sigma_halo(MM, z, CosPar)
    sigma_halo_MM = NFW_sigma_halo(MM, z, CosPar)
    sigma_virial_MM = NFW_approx_sigma_virial(MM, z, CosPar)*np.sqrt(3.) # Use the approximation to speed things up

    sigma_correlation_sqr = np.zeros(nR*nMhalo*nlogM).reshape(nR, nMhalo, nlogM)
    sigma_Mhalo_corr = np.sqrt(1.-sigma_j_M_sqr(0, Mhalo, z, CosPar)**2/sigma_j_M_sqr(1,Mhalo,z,CosPar)/sigma_j_M_sqr(-1,Mhalo,z,CosPar))
    sigma_MM_corr = np.sqrt(1.-sigma_j_M_sqr(0, MM, z, CosPar)**2/sigma_j_M_sqr(1,MM,z,CosPar)/sigma_j_M_sqr(-1,MM,z,CosPar))
    prefix = (100.*CosPar['h']*f_Legendre(0,CosPar))**2
    for j in np.arange(nMhalo):
        sigma_correlation_sqr[:,j,:] = sigma_j_r_M1_M2_sqr(-1, R, Mhalo[j], MM, z, CosPar)*prefix*sigma_Mhalo_corr[j]*sigma_MM_corr
        #for i in np.arange(nR):
        #    sigma_correlation_sqr[i,j,:] = sigma_j_r_M1_M2_sqr(-1, R[i], Mhalo[j], MM, z, CosPar)*prefix*sigma_Mhalo_corr[j]*sigma_MM_corr

    # Integrand = (dn_dlogM*bM*MM/cosmo.rho_critical(z, CosPar)/cosmo.Omega_M_z(z, CosPar)*(sigma_halo_MM**2+
    #            (mu*sigma_virial_MM)**2)).reshape(1,1,nlogM)*(1.+bMhalo.reshape(1, nMhalo, 1)*bM.reshape(1,1,nlogM)*xiR.reshape(nR, nMhalo, 1))
    # Integrand = (dn_dlogM*bM*MM/cosmo.rho_critical(z, CosPar)/cosmo.Omega_M_z(z, CosPar)*((sigma_halo_MM**2+
    #            (mu*sigma_virial_MM)**2).reshape(1,1,nlogM)-2*sigma_correlation_sqr))*(1.+bMhalo.reshape(1, nMhalo, 1)*bM.reshape(1,1,nlogM)*xiR.reshape(nR, nMhalo, 1))
    # sigma2_average = np.sum((Integrand[:,:,2:]+Integrand[:,:,:nlogM-2]+4.*Integrand[:,:,1:nlogM-1])/6.*dlogM, axis=2)#.reshape(nR, nMhalo)

    sigma2_average = (sigma_halo**2).reshape(1,nMhalo)+(sigma_halo_MM**2).reshape(1,nMhalo)+((mu*sigma_virial_MM)**2).reshape(1,nMhalo)-2.*sigma_correlation_sqr.reshape(nR,nMhalo)*1.
    # print 'sigma_correlation_sqr: ', sigma_correlation_sqr.shape
    # print 'sigma2_average: ', sigma_correlation_sqr.shape
    # print 'sigma_halo_MM: ', sigma_halo_MM.shape
    # print sigma_halo_MM
    # print mu*sigma_virial_MM
    # print sigma_correlation_sqr
 
    return np.sqrt(sigma2_average)
    # return np.sqrt((sigma_halo**2).reshape(1,nMhalo)+(sigma_halo_MM**2).reshape(1,nMhalo)+((mu*sigma_virial_MM)**2).reshape(1,nMhalo))


def NFW_project_2h_sigma(y, Mhalo, z, CosPar, fcorr=1., alpha=1.):
    """
    NFW_project_2h_sigma(y, Mhalo, z, CosPar, fcorr=1., alpha=1.):
    """
    # Need to do some argumente/keyword checking
    bbeta = 0.0
    R_min = 1E-5
    R_max = 2E3
    if min(y)<R_min: raise ValueError("y too small")
    if max(y)>(R_max/2.): raise ValueError("y too large")

    y = np.array(y)
    y = y.reshape(y.shape[0], y.size/y.shape[0]) # Convert it to y(ny,1) or y(ny,nMhalo)
    ny = y.shape[0]
    Mhalo = np.array(Mhalo)
    nMhalo = Mhalo.size
    yy = np.ones(nMhalo)*y.reshape(ny,1)

    bMhalo = bias(Mhalo, z, CosPar)
    # Set up interpolation profiles
    dlogR = 1E-2
    R = np.exp(np.arange(np.log(R_min)-2.*dlogR, np.log(R_max)+2.*dlogR, dlogR))
    nR = R.shape[0]
    R = np.ones(nMhalo)*R.reshape(nR, 1) # R(nR, nMhalo)
    xiR = xi_2h_gal_dm(R, 1E13, z, CosPar) # no bias
    #rhoR = NFW_profile(R, Mhalo, z, CosPar, fcorr, alpha)*NFW_sigma(R, Mhalo, z, CosPar, fcorr, alpha)**2 # sigma^2*rhoR(nR, nMhalo)
    #rhoR = (1.+bMhalo*xiR)*NFW_2h_sigma(R, Mhalo, z, CosPar, fcorr, alpha)**2/3. # sigma^2*rhoR(nR, nMhalo)
    #Norm_rhoR = (1.+bMhalo*xiR) # sigma^2*rhoR(nR, nMhalo)
    rhoR = xiR*NFW_2h_sigma(R, Mhalo, z, CosPar, fcorr, alpha)**2/3. # sigma^2*rhoR(nR, nMhalo)
    Norm_rhoR = xiR # sigma^2*rhoR(nR, nMhalo)

    SS_max = R_max/2.0

    Sigma_y = np.zeros((ny, nMhalo)) # Initialize output: Sigma_y(ny, nMhalo)
    Norm_Sigma_y = np.zeros((ny, nMhalo)) # Initialize output: Sigma_y(ny, nMhalo)

    # Very difficult to vectorize without approximation
    # Let's settle for what we have now
    dlogs = 1E-3 # 10% better than 1E-2
    for i in np.arange(nMhalo):
        f = interpolate.interp1d(R[:,i], rhoR[:,i])
        Norm_f = interpolate.interp1d(R[:,i], Norm_rhoR[:,i])
        Sigma_integrand = lambda R, s: (1.-bbeta*s*s/R/R)*2.*f(R)*R**2/np.sqrt(R*R-s*s) # dlogR
        Norm_Sigma_integrand = lambda R, s: 2.*Norm_f(R)*R**2/np.sqrt(R*R-s*s) # dlogR
        # s_max = max(SS_max, max(yy[:,i])*1.1)
        s_max = SS_max
        s_min = yy[:,i]
        for j in np.arange(ny):
            ss = np.exp(np.arange(np.log(s_min[j])+dlogs,np.log(s_max)+dlogs,dlogs))
            nlogs = ss.size
            Integrand = Sigma_integrand(ss, s_min[j])
            Norm_Integrand = Norm_Sigma_integrand(ss, s_min[j])
            Sigma_y[j, i] = np.sum((Integrand[2:]+Integrand[:nlogs-2]+4.*Integrand[1:nlogs-1])/6.*dlogs)*1E6
            Norm_Sigma_y[j, i] = np.sum((Norm_Integrand[2:]+Norm_Integrand[:nlogs-2]+4.*Norm_Integrand[1:nlogs-1])/6.*dlogs)*1E6

    return np.sqrt(Sigma_y/Norm_Sigma_y)

def xi_Legendre_monopole(R, Mhalo, z, CosPar):
    """
    xi_Legendre_monopole(R, Mhalo, z, CosPar):
    """

    Mhalo = np.array(Mhalo)
    nMhalo = Mhalo.size
    bMhalo = bias(Mhalo, z, CosPar)
    f = f_Legendre(z, CosPar)/bMhalo
    coeff = 1.+2./3.*f+0.2*f**2
    xi_R = xi_2h_gal_dm(R, Mhalo, z, CosPar) # no bias

    return coeff*xi_R*bMhalo.reshape(1,nMhalo)

def xi_Legendre_multipole(y, Mhalo, z, CosPar):
    """
    quadrupole, hexadecapole = xi_Legendre_multipole(R, Mhalo, z, CosPar):
    """

    R_min = 1E-5
    R_max = 2E3
    if min(y)<R_min: raise ValueError("y too small")
    if max(y)>R_max: raise ValueError("y too large")

    y = np.array(y)
    y = y.reshape(y.shape[0], y.size/y.shape[0]) # Convert it to y(ny,1) or y(ny,nMhalo)
    ny = y.shape[0]
    Mhalo = np.array(Mhalo)
    nMhalo = Mhalo.size
    yy = np.ones(nMhalo)*y.reshape(ny,1)

    bMhalo = bias(Mhalo, z, CosPar)
    f = f_Legendre(z, CosPar)/bMhalo
    coeff_quadrupole = 4./3.*f+4./7.*f*f
    coeff_hexadecapole = 8./35.*f*f
    xi_y = xi_2h_gal_dm(y, Mhalo, z, CosPar) # no bias

    dlogR = 1E-2
    R = np.exp(np.arange(np.log(R_min)-2.*dlogR, np.log(R_max)+2.*dlogR, dlogR))
    nR = R.shape[0]
    R = np.ones(nMhalo)*R.reshape(nR, 1) # R(nR, nMhalo)
    xi_R = xi_2h_gal_dm(R, Mhalo, z, CosPar) # no bias

    xi_y_bar = np.zeros((ny, nMhalo)) # Initialize output: Sigma_y(ny, nMhalo)
    xi_y_bar2 = np.zeros((ny, nMhalo)) # Initialize output: Sigma_y(ny, nMhalo)

    # Very difficult to vectorize without approximation
    # Let's settle for what we have now
    dlogs = 1E-3 # 10% better than 1E-2
    for i in np.arange(nMhalo):
        f_interpol = interpolate.interp1d(R[:,i], xi_R[:,i])
        xiy_qua_integrand = lambda R: f_interpol(R)*R**3 # dlogR
        xiy_hex_integrand = lambda R: f_interpol(R)*R**5 # dlogR
        s_max = yy[:,i]
        s_min = R_min
        for j in np.arange(ny):
            ss = np.exp(np.arange(np.log(s_min)+dlogs,np.log(s_max[j])+dlogs,dlogs))
            nlogs = ss.size
            qua_Integrand = xiy_qua_integrand(ss)
            hex_Integrand = xiy_hex_integrand(ss)
            xi_y_bar[j, i] = np.sum((qua_Integrand[2:]+qua_Integrand[:nlogs-2]+4.*qua_Integrand[1:nlogs-1])/6.*dlogs)*3./s_max[j]**3
            xi_y_bar2[j, i] = np.sum((hex_Integrand[2:]+hex_Integrand[:nlogs-2]+4.*hex_Integrand[1:nlogs-1])/6.*dlogs)*5/s_max[j]**5

    # Calculate xiR_bar & xiR_bar2
    xiR_quadrupole = coeff_quadrupole*(xi_y-xi_y_bar)
    xiR_hexadecapole = coeff_hexadecapole*(xi_y+2.5*xi_y_bar-3.5*xi_y_bar2)
    return xiR_quadrupole*bMhalo.reshape(1,nMhalo), xiR_hexadecapole*bMhalo.reshape(1,nMhalo)

