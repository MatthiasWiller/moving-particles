"""
# ---------------------------------------------------------------------------
# Several plotting functions
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-05
# ---------------------------------------------------------------------------
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.pyplot import cm
from matplotlib import rcParams
from matplotlib import ticker
from matplotlib.ticker import NullFormatter

from mpl_toolkits.mplot3d import Axes3D

# create figure object with LaTeX font
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 22
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

# -----------------------------------------------------------------------------------------
# histogram plot
def plot_hist(x, target_PDF=0, dimension=1):

    len_x       = len(x)
    n           = np.sqrt(len_x)
    num_bins    = np.math.ceil(n)

    # the histogram of the data
    plt.figure()
    n, bins, patches = plt.hist(x, num_bins, normed=1, color='navy')

    # add a 'best fit' line
    if((target_PDF != 0) and (dimension == 1)):
        # for 1D case
        y = target_PDF(bins)
        plt.plot(bins, y, '--', color='red')

    if((target_PDF != 0) and (dimension == 2)):
        # for 2D case
        y = compute_marginal_PDF(target_PDF, bins, 0)
        plt.plot(bins, y, '--', color='red')

    plt.ylim(0,1)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.title(r'Histogram of $\theta$')
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'Frequency $p$')
    plt.tight_layout()
    #plt.savefig('plot_hist.pdf', format='pdf', dpi=50, bbox_inches='tight')


# -----------------------------------------------------------------------------------------
# plot values of index
def plot_mixing(x):
    
    n_samples = len(x)

    plt.figure()
    plt.plot(x, color='navy')

    plt.title(r'Iterations of the MCMC')
    plt.xlabel(r'Number of samples, $n$')
    plt.ylabel(r'$\theta_1$')
    plt.xlim(0, n_samples)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.tight_layout()
    #plt.savefig('plot_mixing.pdf', format='pdf', dpi=50, bbox_inches='tight')

# -----------------------------------------------------------------------------------------
# plot of the estimated autocorrelation of samples
def plot_autocorr(x, lag):

    # compute sample autocorrelation
    n_samples   = len(x)
    rho         = np.zeros(lag, float)
    sigma2      = x.var()
    x           = x - x.mean()

    for k in range(0, lag):
        temp = 0
        for t in range(0, n_samples - k):
            temp += x[t] * x[t+k]

        rho[k] = (1/sigma2) * (1/(n_samples - k)) * temp
    #rho = hplt.estimate_autocorrelation(x, lag)    

    # plot results
    plt.figure()
    plt.plot(rho, '.')

    plt.title(r'Sample autocorrelation')
    plt.xlabel(r'Lag, $k$')
    plt.ylabel(r'$\hat{R}(k)')
    plt.xlim(0, lag)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tight_layout()
    #plt.savefig('plot_autocorr.pdf', format='pdf', dpi=50, bbox_inches='tight')

# -----------------------------------------------------------------------------------------
# make a nice scatter plot with contour lines of the pdf and samples    
def plot_scatter_with_contour(theta, target_PDF):

    x       = np.linspace(-2, 8, 100)
    X, Y    = np.meshgrid(x, x)
    Z       = target_PDF([X, Y])

    # plot results
    plt.figure()
    plt.contour(X, Y, Z, 7, cmap=cm.jet)
    #plt.scatter(theta[0,:], theta[1,:], marker='o', facecolors='None', color='navy', linewidths=1, label='Circles')
    plt.scatter(theta[0,:], theta[1,:], s=1, color='blue', marker='o', linestyle='None')
    
    #ax.plot_surface(theta[0,:], theta[1,:], target_PDF(theta) )
    plt.title(r'Contour plot')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.xticks([-2, 0, 2, 4, 6, 8])
    plt.yticks([-2, 0, 2, 4, 6, 8])

    plt.gca().set_aspect('equal', adjustable='box')
    #plt.tight_layout()
    #plt.savefig('plot_scatter_with_contour.pdf', format='pdf', dpi=50, bbox_inches='tight')


# -----------------------------------------------------------------------------------------
# plot the surface of a 2D pdf in 3D
def plot_surface_custom(target_PDF):
    x       = np.linspace(-2, 8, 100)
    X, Y    = np.meshgrid(x, x)
    Z       = target_PDF([X, Y])

    plt.figure()
    ax = plt.gca(projection='3d')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.pink)

    ttl = ax.set_title('Target distribution')
    ttl.set_position([.5, 0.95])
    ax.view_init(elev=45, azim=40)
    ax.set_xlim3d(-2, 8)
    ax.set_ylim3d(-2, 8)
    ax.set_zlim3d(0, 0.25)
    ax.xaxis._axinfo['label']['space_factor'] = 20
    ax.yaxis._axinfo['label']['space_factor'] = 20
    ax.zaxis._axinfo['label']['space_factor'] = 20

    plt.tight_layout()
    #plt.savefig('plot3d.pdf', format='pdf', dpi=50, bbox_inches='tight')


# -----------------------------------------------------------------------------------------
# plot combination of Scatter plot with histogram
def plot_scatter_with_hist(x, target_PDF=0):

    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width         = 0.1, 0.65
    bottom, height      = 0.1, 0.65
    bottom_h = left_h   = left + width + 0.02

    # make rectangulars
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure( figsize=(8, 8) )

    # set up scatter and histograms
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot
    axScatter.scatter(x[0, :], x[1, :], marker='o', facecolors='None', \
                      color='navy', linewidths=1, label='Circles')

    # now determine nice limits by hand:
    binwidth = 0.10

    # choose limits of the plot
    lowerlim = -1
    upperlim = 7

    # set limits of scatter plot
    axScatter.set_xlim((lowerlim, upperlim))
    axScatter.set_ylim((lowerlim, upperlim))

    # create bins and plot histograms
    bins = np.arange(lowerlim, upperlim + binwidth, binwidth)
    axHistx.hist(x[0,:], bins=bins, normed=1, color='navy')
    axHisty.hist(x[1,:], bins=bins, orientation='horizontal', normed=1, color='navy')
    
    # plot best-fit line, if target_PDF is given
    if(target_PDF != 0):
        f_x0 = compute_marginal_PDF(target_PDF, bins, 0)
        axHistx.plot(bins, f_x0, '--', color='red')

        f_x1 = compute_marginal_PDF(target_PDF, bins, 1)
        axHisty.plot(f_x1, bins, '--', color='red')
    
    # limit histograms to limits of scatter-plot
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    # tight layout not possible here !
    #plt.savefig('plot_scatter_with_hist.pdf', format='pdf', dpi=50, bbox_inches='tight')

# -----------------------------------------------------------------------------------------
# Helper function to compute the marginal pdf of a 2D multivariate pdf
def compute_marginal_PDF(target_PDF, bins, dimension):
    x_from  = -10
    x_till  = 10
    n_steps = 100.0

    dx      = (x_till - x_from) / n_steps
    x       = np.linspace(x_from, x_till, n_steps)

    len_x       = len(x)
    len_bins    = len(bins)
    y           = np.zeros((len_bins), float)

    # integrate over x1 to obtain marginal of x0
    if dimension == 0:
        for i in range(0, len_bins):
            for j in range(0, len_x):
                temp_x = bins[i]
                temp_y = x[j]
                y[i]  += target_PDF([temp_x, temp_y])

            y[i] = y[i]*dx

    # integrate over x2 to obtain marginal of x1
    if dimension == 1:
        for i in range(0, len_bins):
            for j in range(0, len_x):
                temp_y = bins[i]
                temp_x = x[j]
                y[i]  += target_PDF([temp_x, temp_y])

            y[i] = y[i]*dx

    return y
