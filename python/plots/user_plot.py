"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import plots.help_plot as hplt

from matplotlib.pyplot import *
from matplotlib import rcParams
from matplotlib import ticker
from matplotlib.ticker import NullFormatter

from mpl_toolkits.mplot3d import Axes3D


# -----------------------------------------------------------------------------------------
# histogram plot
def plot_hist(x, target_PDF=0, dimension=1):
    num_bins = 50 # default = 50
    len_x = len(x)
    n = np.sqrt(len_x)
    num_bins = np.math.ceil(n)

    #fig, ax = plt.subplots()

    fig = plt.figure()

    # create figure object with LaTeX font
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 22
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)

    # the histogram of the data
    n, bins, patches = plt.hist(x, num_bins, normed=1)

    # add a 'best fit' line
    if((target_PDF != 0) and (dimension == 1)):
        # for 1D case
        y = target_PDF(bins)
        plt.plot(bins, y, '--')
    
    if((target_PDF != 0) and (dimension == 2)):
        # for 2D case
        y = hplt.compute_marginal_PDF(target_PDF, bins)
        plt.plot(bins, y, '--')
    
    plt.title(r'Histogram of $\theta$')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$p$')


    # Tweak spacing to prevent clipping of ylabel
    plt.tight_layout()
    #plt.savefig('chain_evol.pdf', format='pdf', dpi=50, bbox_inches='tight')


# -----------------------------------------------------------------------------------------
# plot values of index
def plot_mixing(x):
    #fig, ax = plt.subplots()
    fig = plt.figure()

    # create figure object with LaTeX font
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 22
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)

    plt.plot(x)
    
    plt.title(r'Plot of $\theta$')
    plt.xlabel(r'n')
    plt.ylabel(r'$\theta$')


    plt.tight_layout()
    #plt.savefig('chain_evol.pdf', format='pdf', dpi=50, bbox_inches='tight')

# -----------------------------------------------------------------------------------------
# plot of the estimated autocorrelation of samples
def plot_autocorr(x, n_samples):
    
    # number of samples to use
    n = np.minimum(len(x),n_samples)
    x = x[:n]

    # compute autocorrelation estimator
    y = hplt.estimate_autocorrelation(x)    

    #fig, ax = plt.subplots()
    fig = plt.figure()

    # create figure object with LaTeX font
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 22
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)

    # plot results
    plt.stem(y)
    plt.title(r'Estimaed Autocorrelation')
    plt.xlabel(r'n')
    plt.ylabel(r'autocorrelation')

    # Tweak spacing to prevent clipping of ylabel
    plt.tight_layout()
    #plt.savefig('chain_evol.pdf', format='pdf', dpi=50, bbox_inches='tight')

# -----------------------------------------------------------------------------------------
# make a nice scatter plot with contour lines of the pdf and samples    
def plot_scatter_with_contour(theta, target_PDF):
    #fig, ax = plt.subplots()
    fig = plt.figure()
    
    # create figure object with LaTeX font
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 22
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)

    #fig =  plt.figure()
    #ax = fig.gca(projection='3d')

    # plot results
    plt.scatter(theta[0,:], theta[1,:], marker='o', facecolors='None', color='navy', linewidths=1, label='Circles')
    
    #ax.plot_surface(theta[0,:], theta[1,:], target_PDF(theta) )
    plt.title(r'Contour plot')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

    X = np.arange(-2, 8, 0.25)
    xlen = len(X)
    Y = np.arange(-2, 9, 0.25)
    ylen = len(Y)
    X, Y = np.meshgrid(X, Y)
    Z = target_PDF([X,Y])
    plt.contour(X, Y, Z)

    # Tweak spacing to prevent clipping of ylabel
    plt.tight_layout()
    #plt.savefig('chain_evol.pdf', format='pdf', dpi=50, bbox_inches='tight')


# -----------------------------------------------------------------------------------------
# plot the surface of a 2D pdf in 3D
def plot_surface_custom(target_PDF):
    fig = plt.figure()
    plt = fig.gca(projection='3d')

    X = np.arange(-2, 8, 0.25)
    xlen = len(X)
    Y = np.arange(-2, 9, 0.25)
    ylen = len(Y)
    X, Y = np.meshgrid(X, Y)
    Z = target_PDF([X,Y])
    plt.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False)


# -----------------------------------------------------------------------------------------
# plot combination of Scatter plot with histogram
def plot_scatter_hist(x, target_PDF):

    # the random data
    x = np.random.randn(1000)
    y = np.random.randn(1000)

    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    lim = (int(xymax/binwidth) + 1) * binwidth

    axScatter.set_xlim((-lim, lim))
    axScatter.set_ylim((-lim, lim))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
