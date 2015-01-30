
# coding: utf-8

# In[1]:

import absorberspec, ebossspec, sdssspec, datapath, fitsio, starburstspec
import cookb_signalsmooth as cbs
from scipy.stats import nanmean, nanmedian
import cosmology as cosmo
masterwave, allflux, allivar = ebossspec.rest_allspec_readin()
objs_ori = ebossspec.elg_readin()
nobj = objs_ori.size


# In[2]:

index_wave_all = searchsorted(masterwave, [1900., 7200.])
tmpflux = allflux[index_wave_all[0]:index_wave_all[1],:]
tmpivar = allivar[index_wave_all[0]:index_wave_all[1],:]
tmpwave = masterwave[index_wave_all[0]:index_wave_all[1]]


# In[4]:

tmpmedian = zeros(tmpwave.size)
for i in arange((tmpflux.shape)[0]):
    iuse = (where(np.logical_and(tmpivar[i,:]>0, objs_ori['zGOOD']==1)))[0]
    tmpmedian[i] = median(tmpflux[i,iuse])


# In[263]:

nemlines = 15
emission = zeros(nemlines, dtype=[('NAME','S10'),('XPOS','f4'),('YPOS','f4'), ('WAVE','f4')])
emission[0] = ('FeII*', 2344.-0., 0.52, 2344.)
emission[1] = ('FeII*', 2600.-25., 0.52, 2600.)
emission[2] = ('[OII]', 3727.+20., 1.75, 3727.)
emission[3] = ('[NeIII]', 3868.-100., 0.63, 3868.)
#emission[4] = ('[OIII]', 4363., 0.8, 4363.)
emission[5] = ('[OIII]', 4959.-100., 0.96, 4959.)
emission[6] = ('[OIII]', 5007.-50., 1.75, 5007.)
emission[7] = ('HeI', 5876.-55., 0.56, 5876.)
emission[8] = ('[OI]', 6302.-70., 0.51, 6302.)
emission[9] = ('[NII]', 6548.-120., 0.57, 6548.)
emission[10] = ('[NII]', 6583.-5., 0.89, 6583.)
emission[11] = ('[SII]', 6716.+20., 0.93, 6716.)
#emission[12] = ('SII', 6730., 1.2, 6730.)
emission[13] = ('[ArIII]', 7137.-100., 0.45, 7137.)
emission[14] = ('CII]', 2327.34-60., 0.60, 2327.34)
 
nabslines = 10
absorption = zeros(nabslines, dtype=[('NAME','S20'),('XPOS','f4'),('YPOS','f4'),('WAVE', 'f4')])
absorption[0] = ('FeII', 2300.+10., 0.20, 2300.)
absorption[1] = ('FeII', 2600.-60., 0.16, 2600.)
absorption[2] = ('MgII', 2800.-60., 0.17, 2800.)
absorption[3] = ('MgI', 2853.-30., 0.24, 2853.)
absorption[4] = ('CaII', 3934.78-180., 0.05, 3934.78)
absorption[5] = ('K', 3934.78-25., 0.05, 3934.78)
absorption[6] = ('H', 3969.59-10., 0.05, 3969.59)
absorption[7] = ('MgI b', 5183.62-150., 0.15, 5183.62)
absorption[8] = (r'NaI D$_{2,1}$', 5900.-165., 0.15, 5900.)
absorption[9] = (r'G band', 4307.-15., 0.09, 4307.)


nbalmer = 7
balmer = zeros(nbalmer, dtype=[('NAME','S10'),('XPOS','f4'),('YPOS','f4'),('WAVE', 'f4')])
balmer[0] = (r'$\alpha$', 6563.-35., 0.25, 6563.)
balmer[1] = (r'$\beta$', 4861.-35., 0.25, 4861.)
balmer[2] = (r'$\gamma$', 4341.-35., 0.265, 4341.)
balmer[3] = (r'$\delta$', 4102.-35., 0.24, 4102.)
balmer[4] = (r'$\epsilon$', 3970.-30., 0.22, 3907.)
balmer[5] = (r'$\zeta$', 3889.-28., 0.235, 3889.)
balmer[6] = (r'$\eta$', 3835.-30., 0.22, 3835.)


# In[266]:

fig = figure(figsize=(20,8))
ax = fig.add_subplot(111)
ax.plot(tmpwave, tmpmedian)
ax.set_xlim(2000, 7250)
ax.set_ylim(-0.05, 2.0)
ax.set_title('Composite Spectrum of Emission Line Galaxies (ELGs) from eBOSS', fontsize=15)
ax.set_xlabel(r'$\lambda$ [$\AA$]', fontsize=20)
ax.set_ylabel(r'$f(\lambda)$ [arbitrary unit]', fontsize=20)
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2.0)
ax.tick_params(axis='both', which='both', length=7, width=2, labelsize=15)
#emission_name = array(['FeII*', 'FeII*', '[OII]', '[NeIII]','[OIII]', '[OIII]', 'HeI', '[OI]',
#                       'NII', 'NII', 'SII', 'SII', '[ArIII]'])
#emission_xpos = array([2300.,   2550.,   3727.,   3868.,     4363.,   4959.,    5876., 6300.,
#                       6548., 6583., 6716., 6730., 7137.])
for i in arange(nemlines):
    etmp = emission[i]
    ax.text(etmp['XPOS'], etmp['YPOS'], etmp['NAME'], fontsize=14, color='g')
for i in arange(nabslines):
    atmp = absorption[i]
    ax.text(atmp['XPOS'], atmp['YPOS'], atmp['NAME'], fontsize=14)
for i in arange(nbalmer):
    btmp = balmer[i]
    ax.text(btmp['XPOS'], btmp['YPOS'], btmp['NAME'], color='brown', fontsize=20)
plot([3934.78, 3934.78], [0.115, 0.18], 'k') # Ca II
plot([3969.59, 3969.59], [0.115, 0.18], 'k') # Ca II
plot([5891.58, 5891.58], [0.225, 0.30], 'k') # Na I
plot([5897.56, 5897.56], [0.225, 0.30], 'k') # Na I
plot([4307.90, 4307.90], [0.155, 0.22], 'k') # G (Ca, CH+, Fe)
plot([5183.62, 5183.62], [0.225, 0.30], 'k') # Mg I
plot([5172.70, 5172.70], [0.225, 0.30], 'k') # Mg I 
plot([5167.33, 5167.33], [0.225, 0.30], 'k')  # Mg I


# In[7]:

(outwave, fluxmean, fluxmedian, norm_fluxmean, norm_fluxmedian) = ebossspec.feiimgii_composite()


# In[230]:

fig = figure(figsize=(20,7))
this_ylim = [0.15, 1.8]
ax = fig.add_subplot(111)
dwave = median(outwave[1:]-outwave[:-1])
ax.plot(outwave+dwave/2., norm_fluxmedian, 'b', drawstyle='steps', linewidth=1)
yabs = 1.-(1.-absorberstack['FLUXMEDIAN'])*2.
ax.plot(absorberstack['WAVE'], yabs, 'r', linewidth=2)
ax.set_ylim(this_ylim)
ax.set_xlim(2200, 2900)
#ax.set_title('Composite Spectrum of Emission Line Galaxies (ELGs) from eBOSS', fontsize=15)
ax.set_xlabel(r'$\lambda$ [$\AA$]', fontsize=20)
ax.set_ylabel(r'Normalized $f(\lambda)$', fontsize=20)
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2.0)
ax.tick_params(axis='both', which='both', length=7, width=2, labelsize=15)
ax.text(2210, 1.65, 'Blue: Emission Line Galaxies at 0.6<z<1.0 (eBOSS 2015)', color='b', fontsize=16, fontname='serif')
ax.text(2210, 1.55, 'Red:  Quasar Absorption-line Systems at 0.4<z<2.2 (Zhu 2013)', color='r', fontsize=16, fontname='serif')
#plot([2200, 2900], [1,1], '--', color='0.75')
#plot([2320,2320], this_ylim)
#plot([2750,2750], this_ylim)


# In[10]:

absorberstack = (fitsio.read('/Users/Benjamin/AstroData/Absorbers/Absorbers_Composite_Allabs_0.8AA.fits'))[0]


# In[228]:

fig = figure(figsize=(10,7))
this_ylim = [0.15, 1.8]
ax = fig.add_subplot(111)
dwave = median(outwave[1:]-outwave[:-1])
ax.plot(outwave+dwave/2., norm_fluxmedian, 'b', drawstyle='steps', linewidth=1)
ax.set_ylim(this_ylim)
ax.set_xlim(2230, 2410)
#ax.set_title('Composite Spectrum of Emission Line Galaxies (ELGs) from eBOSS', fontsize=15)
ax.set_xlabel(r'$\lambda$ [$\AA$]', fontsize=20)
ax.set_ylabel(r'Normalized $f(\lambda)$', fontsize=20)
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2.0)
ax.tick_params(axis='both', which='both', length=7, width=2, labelsize=15)

this_nabslines = 6
this_absorption = zeros(this_nabslines, dtype=[('NAME','S20'),('XPOS','f4'),('YPOS','f4'),('WAVE', 'f4')])
this_absorption[0] = ('FeII 2249.88', 2249.88-5, 0.2, 2249.88)
this_absorption[1] = ('FeII 2260.78', 2260.78+1.5, 0.2, 2260.78)
this_absorption[2] = ('FeII 2344.21', 2344.21+1.5, 0.2, 2344.21)
this_absorption[3] = ('FeII 2374.46', 2374.46+1.5, 0.2, 2374.46)
this_absorption[4] = ('FeII 2382.76', 2382.76+1.5, 0.2, 2382.76)
this_absorption[5] = ('CIII 2297.58', 2297.58+1.5, 0.2, 2297.58)


this_nemlines = 3
this_emission = zeros(this_nemlines, dtype=[('NAME','S20'),('XPOS','f4'),('YPOS','f4'), ('WAVE','f4')])
this_emission[0] = ('CII] 2327.34', 2327.34+1., 1.3, 2327.34)
this_emission[1] = ('FeII* 2396.36', 2396.36+1., 1.3, 2396.36)
this_emission[2] = ('FeII* 2365.55', 2365.55+1., 1.3, 2365.55)

#this_emission[2] = ('FeII* 2632.11', 2632.11+10., 1.2, 2632.11)

for i in arange(this_nabslines):
    atmp = this_absorption[i]
    ax.text(atmp['XPOS'], atmp['YPOS'], atmp['NAME'], fontsize=14, rotation='vertical',va='bottom')
    ax.plot([atmp['WAVE'], atmp['WAVE']], this_ylim, '--k')
for i in arange(this_nemlines):
    etmp = this_emission[i]
    ax.text(etmp['XPOS'], etmp['YPOS'], etmp['NAME'], color='g', rotation='vertical',va='bottom', fontsize=14)
    ax.plot([etmp['WAVE'], etmp['WAVE']], this_ylim, ':g', linewidth=1)
    
# Replot
ax.plot(outwave+dwave/2., norm_fluxmedian, 'b', drawstyle='steps', linewidth=1)
yabs = 1.-(1.-absorberstack['FLUXMEDIAN'])*2.
ax.plot(absorberstack['WAVE'], yabs, 'r', linewidth=2)


# In[165]:

fig = figure(figsize=(10,7))
this_ylim = [0.15, 1.8]
ax = fig.add_subplot(111)
dwave = median(outwave[1:]-outwave[:-1])
ax.plot(outwave+dwave/2., norm_fluxmedian, 'b', drawstyle='steps', linewidth=1)
ax.set_ylim(this_ylim)
ax.set_xlim(2550, 2650)
#ax.set_title('Composite Spectrum of Emission Line Galaxies (ELGs) from eBOSS', fontsize=15)
ax.set_xlabel(r'$\lambda$ [$\AA$]', fontsize=20)
ax.set_ylabel(r'Normalized $f(\lambda)$', fontsize=20)
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2.0)
ax.tick_params(axis='both', which='both', length=7, width=2, labelsize=15)

this_nabslines = 5
this_absorption = zeros(this_nabslines, dtype=[('NAME','S20'),('XPOS','f4'),('YPOS','f4'),('WAVE', 'f4')])
this_absorption[0] = ('MnII 2576.88', 2576.88-3, 0.2, 2576.88)
this_absorption[1] = ('FeII 2586.65', 2586.65-3, 0.2, 2586.65)
this_absorption[2] = ('MnII 2594.50', 2594.50-3, 0.2, 2594.50)
this_absorption[3] = ('FeII 2600.17', 2600.17+1.5, 0.2, 2600.17)
this_absorption[4] = ('MnII 2606.46', 2606.46+1.5, 0.2, 2606.46)

this_nemlines = 3
this_emission = zeros(this_nemlines, dtype=[('NAME','S20'),('XPOS','f4'),('YPOS','f4'), ('WAVE','f4')])
this_emission[0] = ('FeII* 2626.45', 2626.45+1.5, 1.3, 2626.45)
this_emission[1] = ('FeII* 2612.65', 2612.65+1.5, 1.3, 2612.65)
#this_emission[2] = ('FeII* 2632.11', 2632.11+10., 1.2, 2632.11)

for i in arange(this_nabslines):
    atmp = this_absorption[i]
    ax.text(atmp['XPOS'], atmp['YPOS'], atmp['NAME'], fontsize=14, rotation='vertical',va='bottom')
    ax.plot([atmp['WAVE'], atmp['WAVE']], this_ylim, '--k')
for i in arange(this_nemlines):
    etmp = this_emission[i]
    ax.text(etmp['XPOS'], etmp['YPOS'], etmp['NAME'], color='g', rotation='vertical',va='bottom', fontsize=14)
    ax.plot([etmp['WAVE'], etmp['WAVE']], this_ylim, ':g', linewidth=1)
    
# Replot
ax.plot(outwave+dwave/2., norm_fluxmedian, 'b', drawstyle='steps', linewidth=2)
yabs = 1.-(1.-absorberstack['FLUXMEDIAN'])*2.
ax.plot(absorberstack['WAVE'], yabs, 'r', linewidth=2)


# In[179]:

fig = figure(figsize=(10,7))
this_ylim = [0.15, 1.8]
ax = fig.add_subplot(111)
dwave = median(outwave[1:]-outwave[:-1])
ax.plot(outwave+dwave/2., norm_fluxmedian, 'b', drawstyle='steps', linewidth=1)
ax.set_ylim(this_ylim)
ax.set_xlim(2760, 2880)
#ax.set_title('Composite Spectrum of Emission Line Galaxies (ELGs) from eBOSS', fontsize=15)
ax.set_xlabel(r'$\lambda$ [$\AA$]', fontsize=20)
ax.set_ylabel(r'Normalized $f(\lambda)$', fontsize=20)
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2.0)
ax.tick_params(axis='both', which='both', length=7, width=2, labelsize=15)

this_nabslines = 3
this_absorption = zeros(this_nabslines, dtype=[('NAME','S20'),('XPOS','f4'),('YPOS','f4'),('WAVE', 'f4')])
this_absorption[0] = ('MgII 2796.35', 2796.35-4., 0.2, 2796.35)
this_absorption[1] = ('MgII 2803.53', 2803.53+1.5, 0.2, 2803.53)
this_absorption[2] = ('MgI 2852.96', 2852.96+1.5, 0.21, 2852.96)

for i in arange(this_nabslines):
    atmp = this_absorption[i]
    ax.text(atmp['XPOS'], atmp['YPOS'], atmp['NAME'], fontsize=14, rotation='vertical',va='bottom')
    ax.plot([atmp['WAVE'], atmp['WAVE']], this_ylim, '--k')
#plot([2344.21, 2344.21], [0.25, 1.25], 'g')
#plot([2374.46, 2374.46], [0.25, 1.25], 'g')
#plot([2382.77, 2382.77], [0.25, 1.25], 'g')
#plot([2365.55, 2365.55], [0.25, 1.25], 'r')
#plot([2396.35, 2396.35], [0.25, 1.25], 'r')

# Replot
ax.plot(outwave+dwave/2., norm_fluxmedian, 'b', drawstyle='steps', linewidth=2)
yabs = 1.-(1.-absorberstack['FLUXMEDIAN'])
ax.plot(absorberstack['WAVE'], yabs, 'r', linewidth=2)


# In[169]:

fig = figure(figsize=(12,7))
ax = fig.add_subplot(111)
dwave = median(outwave[1:]-outwave[:-1])
ax.plot(outwave+dwave/2., norm_fluxmedian, 'b', drawstyle='steps', linewidth=1)
this_ylim = [0.15, 2.]
ax.set_ylim(this_ylim)
ax.set_xlim(2980, 4050)
#ax.set_title('Composite Spectrum of Emission Line Galaxies (ELGs) from eBOSS', fontsize=15)
ax.set_xlabel(r'$\lambda$ [$\AA$]', fontsize=20)
ax.set_ylabel(r'Normalized $f(\lambda)$', fontsize=20)
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2.0)
ax.tick_params(axis='both', which='both', length=7, width=2, labelsize=15)

this_nabslines = 8
this_absorption = zeros(this_nabslines, dtype=[('NAME','S20'),('XPOS','f4'),('YPOS','f4'),('WAVE', 'f4')])
this_absorption[0] = ('TiII 3067.24', 3067.24-30., 0.22, 3067.24)
this_absorption[1] = ('TiII 3073.86', 3073.86+10., 0.22, 3073.86)
#this_absorption[2] = ('AlI 3083.05', 3083.05+1.5, 0.23, 3083.05)
this_absorption[3] = ('TiII 3230.12', 3230.12-30., 0.23, 3230.12)
this_absorption[4] = ('TiII 3242.92', 3242.92+10., 0.23, 3242.92)
this_absorption[5] = ('TiII 3384.73', 3384.73+10., 0.23, 3384.73)
this_absorption[6] = ('CaII 3934.78', 3934.78-30., 0.23, 3934.78)
this_absorption[7] = ('CaII 3969.59', 3969.59+10., 0.23, 3969.59)

this_nemlines = 3
this_emission = zeros(this_nemlines, dtype=[('NAME','S20'),('XPOS','f4'),('YPOS','f4'), ('WAVE','f4')])
this_emission[0] = ('[OII] 3727.1', 3727.1-35., 1.4, 3727.1)
this_emission[1] = ('[OII] 3729.8', 3729.8+10., 1.4, 3729.9)
this_emission[2] = ('[NeIII] 3869.77', 3869.77+10., 1.4, 3869.77)

this_nbalmer = 10
this_balmer = zeros(this_nbalmer, dtype=[('NAME','S10'),('XPOS','f4'),('YPOS','f4'),('WAVE', 'f4')])
#this_balmer[0] = (r'$\alpha$', 6563.-35., 0.25, 6563.)
#this_balmer[1] = (r'$\beta$', 4861.-35., 0.25, 4861.)
#this_balmer[2] = (r'$\gamma$', 4341.-35., 0.265, 4341.)
#this_balmer[3] = (r'$\delta$', 4102.-35., 0.24, 4102.)
this_balmer[4] = (r'$\epsilon$', 3970.+10., 0.9, 3907.)
this_balmer[5] = (r'$\zeta$', 3889.-10., 0.85, 3889.)
this_balmer[6] = (r'$\eta$', 3835.38-10., 0.75, 3835.38)
this_balmer[7] = (r'$\theta$', 3797.90-10., 0.8, 3797.90)
this_balmer[8] = (r'$\iota$', 3770.63-7., 0.82, 3770.63)
this_balmer[9] = (r'$\kappa$', 3750.15-10., 0.77, 3750.15)

for i in arange(this_nabslines):
    atmp = this_absorption[i]
    ax.text(atmp['XPOS'], atmp['YPOS'], atmp['NAME'], fontsize=14, rotation='vertical',va='bottom')
    ax.plot([atmp['WAVE'], atmp['WAVE']], this_ylim, '--k')
for i in arange(this_nbalmer):
    btmp = this_balmer[i]
    ax.text(btmp['XPOS'], btmp['YPOS'], btmp['NAME'], color='brown', fontsize=20)
for i in arange(this_nemlines):
    etmp = this_emission[i]
    ax.text(etmp['XPOS'], etmp['YPOS'], etmp['NAME'], color='g', rotation='vertical',va='bottom', fontsize=14)
    ax.plot([etmp['WAVE'], etmp['WAVE']], this_ylim, ':g', linewidth=1)

   
# Replot    
ax.plot(outwave+dwave/2., norm_fluxmedian, 'b', drawstyle='steps', linewidth=1)
yabs = 1.-(1.-absorberstack['FLUXMEDIAN'])*20.+0.26
ax.plot(absorberstack['WAVE'], yabs, 'r', linewidth=2)


# In[200]:

reload(starburstspec)


# In[201]:

(sboutwave, sbfluxmean, sbfluxmedian, sbfluxused) = starburstspec.mgii_composite()


# In[226]:

fig = figure(figsize=(20,7))
this_ylim = [0.15, 1.8]
ax = fig.add_subplot(111)
dwave = median(outwave[1:]-outwave[:-1])
ax.plot(outwave+dwave/2., norm_fluxmedian, 'b', drawstyle='steps', linewidth=2)
ax.set_ylim(this_ylim)
ax.set_xlim(2200, 2900)
#ax.set_title('Composite Spectrum of Emission Line Galaxies (ELGs) from eBOSS', fontsize=15)
ax.set_xlabel(r'$\lambda$ [$\AA$]', fontsize=20)
ax.set_ylabel(r'Normalized $f(\lambda)$', fontsize=20)
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2.0)
ax.tick_params(axis='both', which='both', length=7, width=2, labelsize=15)
ax.text(2210, 1.65, 'Blue: Emission Line Galaxies at 0.6<z<1.0 (eBOSS 2015)', color='b', fontsize=16, fontname='serif')
ax.text(2210, 1.55, 'Red:  Star-burst regions at z~0 (Leitherer 2011)', color='r', fontsize=16, fontname='serif')
ax.plot(sboutwave, sbfluxmedian, 'r', linewidth=2)
#plot([2299., 2299.], this_ylim, 'k')
plot([2297., 2297.], [0.6,0.85], 'k')


# In[249]:

fig = figure(figsize=(20,8))
ax = fig.add_subplot(111)
ax.plot(tmpwave, tmpmedian)
ax.set_xlim(5000, 5500)
ax.set_ylim(0.2, 0.5)
ax.set_title('Composite Spectrum of Emission Line Galaxies (ELGs) from eBOSS', fontsize=15)
ax.set_xlabel(r'$\lambda$ [$\AA$]', fontsize=20)
ax.set_ylabel(r'$f(\lambda)$ [arbitrary unit]', fontsize=20)
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2.0)
ax.tick_params(axis='both', which='both', length=7, width=2, labelsize=15)
#emission_name = array(['FeII*', 'FeII*', '[OII]', '[NeIII]','[OIII]', '[OIII]', 'HeI', '[OI]',
#                       'NII', 'NII', 'SII', 'SII', '[ArIII]'])
#emission_xpos = array([2300.,   2550.,   3727.,   3868.,     4363.,   4959.,    5876., 6300.,
#                       6548., 6583., 6716., 6730., 7137.])
plot([5183.62,5183.62],[0.2,0.5])
plot([5172.70,5172.70],[0.2,0.5])
plot([5167.33,5167.33],[0.2,0.5])
plot([5200.33,5200.33],[0.2,0.5])


# In[ ]:



