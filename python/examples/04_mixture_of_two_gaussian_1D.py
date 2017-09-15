"""
# ---------------------------------------------------------------------------
# Metropolis-Hastings algorithm 1D example: Mixture of two Gaussian
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
from matplotlib.ticker import NullFormatter

import scipy.stats as scps

import utilities.plots as uplt
import algorithms.metropolis_hastings as mh



# INPUT 

np.random.seed(0)

# target pdf (two weighted gaussian)
mu          = [0.0, 10.0] 
sigma       = [2.0, 2.0]
w           = [0.3, 0.7]
target_PDF  = lambda x: w[0]*scps.norm.pdf(x, mu[0], sigma[0]) + w[1]*scps.norm.pdf(x, mu[1], sigma[1])

# proposal pdf 
sigma_prop      = 10
sample_prop_PDF = lambda param: np.random.normal(param, sigma_prop, 1)
f_prop_PDF      = lambda x, param: scps.norm.pdf(x, param, sigma_prop)


#initial_theta  = 20*np.random.uniform(-1.0, 1.0, 1)         # initial theta
initial_theta  = 0
n_samples      = 1000                                       # number of samples
burnInPeriod   = 0                                        # defines burning-in-period of samples
lagPeriod      = 10                                          # only log every n-th value

# apply MCMC
theta = mh.metropolis_hastings(initial_theta, n_samples, target_PDF, sample_prop_PDF, f_prop_PDF, burnInPeriod, lagPeriod)

# OUTPUT
# ----------------------------------------------------------
# plot histogram
matplotlib.rcParams['font.size'] = 26


x           = theta[0,:]
len_x       = len(x)
n           = np.sqrt(len_x)
num_bins    = np.math.ceil(n)

# the histogram of the data
plt.figure()
n, bins, patches = plt.hist(x, num_bins, normed=1, color='navy')

# add a 'best fit' line
x_bins = np.arange(-10,20,0.05)
y = target_PDF(x_bins)
plt.plot(x_bins, y, '-', color='red')

# limits of axis
plt.ylim(0,.15)
plt.yticks([0, 0.05, 0.10, 0.15])

plt.xlim(-10,20)
plt.xticks([-10, 0, 10, 20])

# set labels
plt.xlabel(r'$x$')
plt.ylabel(r'Frequency $p$')
plt.tight_layout()

plt.savefig('plot_hist.pdf', format='pdf', dpi=50, bbox_inches='tight')



# ----------------------------------------------------------
# plot mixing with hist
x = theta[0, :]
len_x       = len(x)
n           = np.sqrt(len_x)
num_bins    = np.math.ceil(n)

# the histogram of the data
nullfmt = NullFormatter()         # no labels

# definitions for the axes
left, width         = 0.1, 0.65
bottom, height      = 0.1, 0.65
bottom_h = left_h   = left + width + 0.02

# make rectangulars
rect_mixing = [left, bottom, width, height]
rect_hist = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure( figsize=(8, 8) )

# set up scatter and histograms
axMixing = plt.axes(rect_mixing)
axHist = plt.axes(rect_hist)

# no labels
# axHist.xaxis.set_major_formatter(nullfmt)
axHist.yaxis.set_major_formatter(nullfmt)

# the mixing plot
axMixing.plot(x, color='navy')

# now determine nice limits by hand:
binwidth = 0.5

# choose limits of the plot
lowerlim = -10
upperlim = 20

# set limits of mixing
axMixing.set_xlim((0, n_samples))
axMixing.set_ylim((lowerlim, upperlim))

# create bins and plot histograms
bins = np.arange(lowerlim, upperlim + binwidth, binwidth)
axHist.hist(x, bins=bins, orientation='horizontal', normed=1, color='navy')

# plot best-fit line
axHist.plot(target_PDF(bins), bins, '-', color='red')

# limit histograms to limits of mixing-plot
axHist.set_ylim(axMixing.get_ylim())
axHist.set_xlim([0, 0.2])

axMixing.set_xticks([0, 200, 400, 600, 800])
axHist.set_xticks([0, 0.1])

# set labels
axMixing.set_ylabel(r'$x$')
axMixing.set_xlabel(r'Samples $n$')

axHist.set_xlabel(r'$f_X(x)$')

# plt.tight_layout()
plt.savefig('plot_mixing_with_hist.pdf', format='pdf', dpi=50, bbox_inches='tight')


# plot samples
# uplt.plot_hist(theta[0, :], target_PDF)
# uplt.plot_mixing(theta[0, :])
uplt.plot_autocorr(theta[0, :], 20)

plt.show()
