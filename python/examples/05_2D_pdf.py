"""
# ---------------------------------------------------------------------------
# Metropolis-Hastings algorithm 2D example: Example 6.4 Ref. [1]
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-05
# ---------------------------------------------------------------------------
# References:
# 1."Simulation and the Monte Carlo method"
#    Rubinstein and Kroese (2017)
# ---------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scps

import plots.user_plot as uplt
import algorithms.metropolis_hastings as mh

# INPUT

np.random.seed(1)


# target pdf {f(x,y) = 1/20216.335877 * exp(-(x*x*y*y + x*x + y*y - 8*x - 8*y)/2)}
c = 1/20216.335877
target_PDF = lambda x: c * np.exp(-(x[0]*x[0]*x[1]*x[1] + x[0]*x[0] + x[1]*x[1] - 8*x[0] - 8*x[1])/2)

# parameters for proposal pdf
sig = 2
mu  = [0, 0]
cov = [[sig**2, 0],[0, sig**2]]

# proposal pdf
sample_prop_PDF = lambda param: scps.norm.rvs(mu, sig, 2) # draw the components independently
f_prop_PDF      = lambda x, param: scps.multivariate_normal.pdf(x, mu, cov) # PDF


initial_theta   = [1.0, 1.0]     # initial theta
n_samples       = int(1e4)       # number of samples
burnInFraction  = 0.2            # defines burn-in-period of samples
lagPeriod       = 1              # only log every n-th value


# apply mcmc-method
theta = mh.metropolis_hastings(initial_theta, n_samples, target_PDF, sample_prop_PDF, f_prop_PDF, burnInFraction, lagPeriod)

# OUTPUT
print('E[X1] =', round(theta[0,:].mean(), 5))
print('E[X2] =', round(theta[1,:].mean(), 5))

# plot samples
uplt.plot_hist(theta[0,:], target_PDF, 2)
uplt.plot_scatter_with_contour(theta, target_PDF)
uplt.plot_mixing(theta[0, :1000])
uplt.plot_autocorr(theta[0,:], 400)
uplt.plot_scatter_with_hist(theta[:, :5000], target_PDF)
plt.show()
