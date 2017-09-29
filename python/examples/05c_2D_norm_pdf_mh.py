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

import utilities.plots as uplt
import algorithms.mcmc_metropolis_hastings as mh

# INPUT

np.random.seed(1)


# target pdf 
mu1 = np.array([1, 1])
rho1 = 0.6
sig1 = np.array([np.array([1**2, rho1*1*2 ]),np.array([rho1*1*2,2**2])])
mu2 = np.array([3, 1.5])
rho2 = -0.5
sig2 = np.array([np.array([0.7**2,rho2*0.7*0.3]),np.array([rho2*0.7*0.3,0.3**2])])

target_PDF = lambda x: scps.multivariate_normal.pdf(x, mu1, sig1) \
                     + scps.multivariate_normal.pdf(x, mu2, sig2)


# parameters for proposal pdf
rho_p = -0.5
cov_p = [[1**2, rho_p*1*1],[rho_p*1*1, 1**2]]

# proposal pdf
sample_prop_PDF = lambda x: scps.multivariate_normal.rvs(x, cov_p) # PDF
f_prop_PDF      = lambda x, param: scps.multivariate_normal.pdf(x, param, cov_p) # PDF

initial_theta   = [3, 1.5]     # initial theta
n_samples       = int(3e4)       # number of samples
burnInPeriod    = 3000            # defines burn-in-period of samples
lagPeriod       = 20              # only log every n-th value

#test configuration
n_samples = 1000
burnInPeriod   = 100
lagPeriod      = 1


# apply mcmc-method
theta = mh.metropolis_hastings(initial_theta, n_samples, target_PDF, sample_prop_PDF, f_prop_PDF, burnInPeriod, lagPeriod)

# OUTPUT
print('E[X1] =', round(theta[0,:].mean(), 5))
print('E[X2] =', round(theta[1,:].mean(), 5))

np.save('python/data/samples_2D_norm_pdf_mh.npy', theta)

# plot samples
#uplt.plot_hist(theta[0,:], target_PDF, 2)
#uplt.plot_scatter_with_contour(theta, target_PDF)
# uplt.plot_mixing(theta[0, :1000])
# uplt.plot_scatter_with_hist(theta[:, :5000], target_PDF)
# uplt.plot_autocorr(theta[0,:], 50, 1)
# uplt.plot_autocorr(theta[1,:], 50, 2)
# plt.show()
