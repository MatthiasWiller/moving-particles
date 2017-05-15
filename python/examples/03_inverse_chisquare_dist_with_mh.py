"""
# ---------------------------------------------------------------------------
# Metropolis-Hastings algorithm 1D example: Example 3 Ref. [1]
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
# 1."Markov Chain Monte Carlo and Gibbs Sampling"
#    B. Walsh (2004)
# ---------------------------------------------------------------------------
"""

import numpy as np
import scipy.stats as scps
import matplotlib.pyplot as plt

import plots.user_plot as uplt
import algorithms.metropolis_hastings as mh

# INPUT 

# set seed for randomization
np.random.seed(0) 

# target pdf (same target as in 02_inverse_chisquare)
C               = 1.0
df_target       = 5.0
#target_PDF      = lambda x: C * x**(-df_target/2) * np.exp(-4.0/(2*x))
target_PDF      = lambda x: scps.chi2.pdf(x, df_target)

# proposal pdf (chisquare function with df = 2 and df = 10)
df_prop = 2.0
f_prop_PDF      = lambda x, param: scps.chi2.pdf(x, df_prop)
sample_prop_PDF = lambda param: scps.chi2.rvs(df_prop, 1)


initial_theta   = 1.0         # initial theta
n_samples       = 1000        # number of samples
burnInFraction  = 0.0         # defines burn-in-period of samples
lagPeriod       = 1           # only log every n-th value

# apply MCMC
theta = mh.metropolis_hastings(initial_theta, n_samples, target_PDF, sample_prop_PDF, f_prop_PDF, burnInFraction, lagPeriod)

# OUTPUT

# plot samples
uplt.plot_hist(theta[0,:], target_PDF, 1)
uplt.plot_mixing(theta[0,:])
uplt.plot_autocorr(theta[0,:], 400)

plt.show()
