"""
To do:
points outside a contour line (or a polygon)
contour line width
"""
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Test Data
# mean = [0,0]; cov = [[1,0.6],[0.6,1]]
# x,y = np.random.multivariate_normal(mean,cov,5000).T
# xbin = [-4,4]; ybin = [-3,3]
# import quicklook
# quicklook.plot_mean(x,y,xbin=xbin,ybin=ybin)


def plot_mean(x0, y0, xbin=None, ybin=None, nbin=10, statistic='mean'):
    x = np.ravel(x0)
    y = np.ravel(y0)
    if xbin is None:
       xbin = [x.min(), x.max()]
    if ybin is None:
       ybin = [y.min(), y.max()]

    index = (np.where(np.all([x>xbin[0], x<xbin[1], y>ybin[0], y<ybin[1]], axis=0)))[0]
    x = x[index]
    y = y[index]

    # 1D
    statistics = stats.binned_statistic(x, y, statistic=statistic, bins=nbin, range=xbin)
    moments = (stats.binned_statistic(x, y, statistic=np.std, bins=nbin, range=xbin))[0]
    xerr = (statistics[1][1]-statistics[1][0])/2.
    x_center = statistics[1][:nbin]+xerr

    # 2D
    xedge_2d, yedge_2d = np.mgrid[xbin[0]:xbin[1]:31j, ybin[0]:ybin[1]:31j]
    k = stats.kde.gaussian_kde((x,y))
    zi = (k(np.vstack([xedge_2d.ravel(), yedge_2d.ravel()]))).reshape(xedge_2d.shape)
    #print xedge_2d.shape, yedge_2d.shape, zi.shape

    #count2d = np.histogram2d(x, y, range=[xbin, ybin])

    #bins = np.linspace(xbin[0], xbin[1], nbin+1)
    #digitized = np.digitize(y, bins)
    #bin_means = [y[digitized == i].mean() for i in np.arange(nbin)]
    #print statistics[0]
    #print bin_means

    pearson_c, pearson_p = stats.pearsonr(x, y)
    spearman_c, spearman_p = stats.spearmanr(x, y)
    kendall_c, kendall_p = stats.kendalltau(x, y)
    linfit = np.polyfit(x, y, 1)
    poly = np.poly1d(linfit)


    print "Pearson Coeffiecent = {0:.5f}".format(pearson_c)
    print "Spearman Coeffiecent = {0:.5f}".format(spearman_c)
    print r'Kendall $\tau$ = {0:.5f}'.format(kendall_c)
    print 'Best-fit slope = {0:.5f}'.format(linfit[0])

    plt.figure(figsize=(8,10))
    plt.clf()
    ax1 = plt.subplot2grid((2,1),(0,0), rowspan=1)
    plt.subplots_adjust(left=0.10, bottom=0.1, hspace=0)
    plt.plot(x,y,'b+', zorder=1)
    plt.xlim(xbin)
    plt.ylim(ybin)
    # ax1.errorbar(x, y, xerr=asymmetric_error, fmt='o')
    plt.text(xbin[0]+(xbin[1]-xbin[0])/20., ybin[1]-(ybin[1]-ybin[0])/20., "Pearson Coeffiecent = {0:.5f}".format(pearson_c), 
             color='r', fontsize=15, zorder=7)
    plt.text(xbin[0]+(xbin[1]-xbin[0])/20., ybin[1]-(ybin[1]-ybin[0])/20.*2.5, "Spearman Coeffiecent = {0:.5f}".format(spearman_c), 
             color='r', fontsize=15, zorder=8) 
    plt.text(xbin[0]+(xbin[1]-xbin[0])/20., ybin[1]-(ybin[1]-ybin[0])/20.*4., r'Kendall $\tau$ = {0:.5f}'.format(kendall_c), 
             color='r', fontsize=15, zorder=9) 
    plt.errorbar(x_center, statistics[0], xerr=xerr, yerr=moments, ms=4, linewidth=4, ecolor='r', color='r', zorder=10)
    plt.setp(ax1.get_xticklabels(), visible=False)
    #plt.xlabel('x')
    #plt.ylabel('y', rotation='horizontal')

    ax2 = plt.subplot2grid((2,1),(1,0), rowspan=1)
    plt.subplots_adjust(left=0.10, bottom=0.1, hspace=0)
    #plt.hexbin(x,y, gridsize=30)
    plt.pcolormesh(xedge_2d,yedge_2d,zi)
    plt.contour(xedge_2d,yedge_2d,zi, linewidth=6)
    plt.plot(xbin, poly(xbin), 'g', linewidth=4, zorder=10)
    plt.xlim(xbin)
    plt.ylim(ybin)
    plt.show()

    return None

# Useful posts from stackoverflow
############################################
#np.random.seed(1977)
## Generate 200 correlated x,y points
#data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], 200)
#x, y = data.T
#nbins = 20
#fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)
#axes[0, 0].set_title('Scatterplot')
#axes[0, 0].plot(x, y, 'ko')
#axes[0, 1].set_title('Hexbin plot')
#axes[0, 1].hexbin(x, y, gridsize=nbins)
#axes[1, 0].set_title('2D Histogram')
#axes[1, 0].hist2d(x, y, bins=nbins)
## Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
#k = stats.kde.gaussian_kde(data.T)
#xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
#zi = k(np.vstack([xi.flatten(), yi.flatten()]))
#axes[1, 1].set_title('Gaussian KDE')
#axes[1, 1].pcolormesh(xi, yi, zi.reshape(xi.shape))
#fig.tight_layout()
#plt.show()
############################################
