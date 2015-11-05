import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import speclines
import ebossspec, ebossanalysis

data = ebossanalysis.unify_emissionline_profile_readin()
absorption = ebossanalysis.unify_absorptionline_profile_readin()

vel = data['VEL']
yabs = 1.-(1.-absorption['UNIFIEDABSORPTION'])*1.25
yabs[yabs<0.] = 0.
yemi = data['UNIFIEDFLUX']-1.

CM = plt.get_cmap('jet_r')

fig, ax = plt.subplots(figsize=(11,5), ncols=1, nrows=1)
fig.subplots_adjust(hspace=0, top=0.90, bottom=0.15)

nplot = 35
demi = 0.7/nplot
xlimits = [-700, 500]

for i in np.arange(nplot)*demi:
    thiscolor = CM(i/0.7+0.08/0.7)
    flux = yabs+yemi*i 
    flux_norm = np.sum(1.-flux[100-5:100+4])
    flux = 1.-(1.-flux)/flux_norm

    ax.plot(vel, flux, color=thiscolor, lw=3)#, drawstyle='steps')
    ax.set_xlim(-700, 500)
    ax.set_ylim(0.745, 1.03)
    ax.plot(xlimits, [1,1], ':', color='black', lw=2)
    ax.set_xlabel(r'Velocity [km$\,$s$^{-1}$]', fontsize=24)
    ax.set_ylabel(r'$\left<R(\lambda)\right>$', fontsize=24)
    ax.tick_params(axis='x', which='major', length=8, width=2, labelsize=22, pad=8)
    ax.tick_params(axis='y', which='major', length=8, width=2, labelsize=22, pad=8)
    ax.plot([0,0],[-0.01,1.05],':', color='black', lw=2)
    outfile = ('Figures/Movies/Emission_Infill_Movie_{0:4.2f}.jpg').format(i)
    fig.savefig(outfile)
    ax.cla()

#ax.plot(xtmp, ytmp, '--', lw=6, color='green')
#ax.text(-750, 1.3, 'Unified Absorption Profile (Flipped)', color='Red', fontsize=21)
#ax.text(-750, 1.26, 'Unified Emission Profile', color='Blue', fontsize=21)
#ax.text(-750, 1.22, r'Gaussian ($\sigma=108\,$km s$^{-1}$)', color='Green', fontsize=21)
#plt.setp(ax.get_xticklabels(), visible=False)

#fig.savefig('/Users/Benjamin/Dropbox/Zhu_Projects/Fine Structure Emission/Version 1/OIII_Emission_Gaussian.eps')
