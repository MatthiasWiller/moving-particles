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

    fig = plt.figure()

    # create figure object with LaTeX font
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 22
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)

    # the histogram of the data
    n, bins, patches = plt.hist(x, num_bins, normed=1, color='navy')

    # add a 'best fit' line
    if((target_PDF != 0) and (dimension == 1)):
        # for 1D case
        y = target_PDF(bins)
        plt.plot(bins, y, '--', color='red')
    
    if((target_PDF != 0) and (dimension == 2)):
        # for 2D case
        y = hplt.compute_marginal_PDF(target_PDF, bins, 0)
        plt.plot(bins, y, '--', color='red')
    
    plt.title(r'Histogram of $\theta$')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$p$')


    # Tweak spacing to prevent clipping of ylabel
    plt.tight_layout()
    #plt.savefig('plot_hist.pdf', format='pdf', dpi=50, bbox_inches='tight')


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

    plt.plot(x, color='navy')
    
    plt.title(r'Plot of $\theta$')
    plt.xlabel(r'n')
    plt.ylabel(r'$\theta$')


    plt.tight_layout()
    #plt.savefig('plot_mixing.pdf', format='pdf', dpi=50, bbox_inches='tight')

# -----------------------------------------------------------------------------------------
# plot of the estimated autocorrelation of samples
def plot_autocorr(x, lag):
    
    # number of samples to use
    #n = np.minimum(len(x), n_samples)
    #x = x[:n] # use first n samples

    # compute autocorrelation estimator
    y = hplt.estimate_autocorrelation(x, lag)    

    #fig, ax = plt.subplots()
    fig = plt.figure()

    # create figure object with LaTeX font
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 22
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)

    # plot results
    markerline, stemlines, _ = plt.stem(y, color='navy')
    plt.setp(markerline, color='navy')
    plt.setp(stemlines, color='navy')

    plt.title(r'Estimated Autocorrelation')
    plt.xlabel(r'n')
    plt.ylabel(r'autocorrelation')

    # Tweak spacing to prevent clipping of ylabel
    plt.tight_layout()
    #plt.savefig('plot_autocorr.pdf', format='pdf', dpi=50, bbox_inches='tight')

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
    Z = target_PDF([X, Y])
    plt.contour(X, Y, Z)

    # Tweak spacing to prevent clipping of ylabel
    plt.tight_layout()
    #plt.savefig('plot_scatter_with_contour.pdf', format='pdf', dpi=50, bbox_inches='tight')


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
def plot_scatter_with_hist(x, target_PDF=0):

    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    # make rectangulars
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure( figsize=(8, 8) )

    # create figure object with LaTeX font
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 22
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)

    # set up scatter and histograms
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot
    axScatter.scatter(x[0, :], x[1, :], marker='o', facecolors='None', color='navy', linewidths=1, label='Circles')

    # now determine nice limits by hand:
    binwidth = 0.10
    xymax = np.max([np.max(np.fabs(x[0,:])), np.max(np.fabs(x[1,:]))])
    # compute limits of the plot
    #lim = (int(xymax/binwidth) + 1) * binwidth
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
        f_x0 = hplt.compute_marginal_PDF(target_PDF, bins, 0)
        axHistx.plot(bins, f_x0, '--', color='red')

        f_x1 = hplt.compute_marginal_PDF(target_PDF, bins, 1)
        axHisty.plot(f_x1, bins, '--', color='red')
    
    # limit histograms to limits of scatter-plot
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    # tight layout not possible here !

    #plt.savefig('plot_scatter_with_hist.pdf', format='pdf', dpi=50, bbox_inches='tight')
