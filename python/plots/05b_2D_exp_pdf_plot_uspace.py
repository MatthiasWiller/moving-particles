"""
# ---------------------------------------------------------------------------
# Plot exponential PDF in U-space
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-10
# ---------------------------------------------------------------------------
"""

import time as timer

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scps

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.pyplot import cm
from matplotlib import rcParams
from matplotlib import ticker
from matplotlib.ticker import NullFormatter

from mpl_toolkits.mplot3d import Axes3D

import utilities.plots as uplt

np.random.seed(0)

savepdf = True

# load data
proposal = 'cs'
# proposal = 'mh'
theta_x = np.load('python/data/samples_2D_exp_pdf_'+ proposal +'.npy')

# define transformation X2U
phi     = lambda x: scps.norm.cdf(x)
phi_inv = lambda x: scps.norm.ppf(x)

CDF     = lambda x: np.array([1-np.exp(-x[0]), 1-np.exp(-x[1])])
CDF_inv = lambda x: np.array([-np.log(1-x[0]), -np.log(1-x[1])])

transform_U2X = lambda u: CDF_inv(phi(u))
transform_X2U = lambda x: phi_inv(CDF(x))

# define target_PDF
c = 0 # correlation rho = 0
# c = 1 # correlation rho = -0.40
target_PDF = lambda x: (1 - c*(1-x[0]-x[1])+c*c*x[0]*x[1])*np.exp(-(x[0]+x[1]+c*x[0]*x[1]))
target_PDF_u = lambda x: scps.multivariate_normal.pdf(x,[0,0])


# transform samples in u space
theta_u = transform_X2U(theta_x)

# PLOT SCATTER WITH HISTOGRAMS ----------------------------------------------------------------
x = theta_u[:, :5000]
nullfmt = NullFormatter()         # no labels

n_grid = 100
xx       = np.linspace(-5, 5, n_grid)
X, Y    = np.meshgrid(xx, xx)
Z = np.zeros((n_grid, n_grid))
for i in range(0, len(X)):
    for j in range(0, len(Y)):
        Z[i,j] = target_PDF_u(np.array([X[i,j], Y[i,j]]))


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

# the contour
axScatter.contour(X, Y, Z, 5, cmap=cm.jet)

# now determine nice limits by hand:
binwidth = 0.10

# choose limits of the plot
lowerlim = -8
upperlim = 8

# set limits of scatter plot
axScatter.set_xlim((lowerlim, upperlim))
axScatter.set_ylim((lowerlim, upperlim))

# create bins and plot histograms
bins = np.arange(lowerlim, upperlim + binwidth, binwidth)
axHistx.hist(x[0, :], bins=bins, normed=1, color='navy')
axHisty.hist(x[1, :], bins=bins, orientation='horizontal', normed=1, color='navy')

# plot best-fit line, if target_PDF is given
f_x0 = uplt.compute_marginal_PDF(target_PDF_u, bins, 0)
axHistx.plot(bins, f_x0, '--', color='red')

f_x1 = uplt.compute_marginal_PDF(target_PDF_u, bins, 1)
axHisty.plot(f_x1, bins, '--', color='red')

# limit histograms to limits of scatter-plot
axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

# set labels
axHistx.set_ylabel(r'$\phi(u_2)$')
axHisty.set_xlabel(r'$\phi(u_1)$')

axScatter.set_xlabel(r'$u_1$')
axScatter.set_ylabel(r'$u_2$')

# set ticks and limits
axHistx.set_yticks([0, 0.5])
axHistx.set_ylim([0, 0.8])

axHisty.set_xticks([0,0.5])
axHisty.set_xlim([0,0.8])

axScatter.set_xticks([-5,0,5])
axScatter.set_yticks([-5,0,5])

if savepdf:
    # tight layout not possible here !
    plt.savefig('plot_scatter_with_hist_u.pdf', format='pdf', dpi=50, bbox_inches='tight')


plt.show()