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
import matplotlib.pyplot as plt
import scipy.stats as scps

import plots.user_plot as uplt
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


initial_theta  = 20*np.random.uniform(-1.0, 1.0, 1)         # initial theta
n_samples      = 1000                                       # number of samples
burnInFraction = 0.2                                        # defines burning-in-period of samples
lagPeriod      = 1                                          # only log every n-th value

# apply MCMC
theta = mh.metropolis_hastings(initial_theta, n_samples, target_PDF, sample_prop_PDF, f_prop_PDF, burnInFraction, lagPeriod)

# OUTPUT

# plot samples
uplt.plot_hist(theta[0,:], target_PDF)
uplt.plot_mixing(theta[0,:])
uplt.plot_autocorr(theta[0, :], 400)

plt.show()
