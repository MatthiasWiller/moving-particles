"""
# ---------------------------------------------------------------------------
# Metropolis algorithm 1D example: Example 2 Ref. [1]
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
import matplotlib.pyplot as plt

import utilities.plots as uplt
import algorithms.metropolis as ma

# INPUT 

# set seed for randomization
np.random.seed(0) 

# target pdf (scaled inverse chi2)
C                   = 1.0
df                  = 5.0
target_PDF          = lambda x: C* x **(-df/2) * np.exp(-4.0/(2*x))
#target_PDF          = lambda x: scps.chi2.pdf(x, df_target)

# proposal pdf (Uniform)
proposal_PDF        = lambda: np.random.uniform(0, 20, 1)

initial_theta       = 1.0           # initial theta
n_samples           = 10000         # number of samples
burningInFraction   = 0.2           # defines burning-in-period of samples
lagPeriod           = 1             # only log every n-th value

# apply MCMC
theta = ma.metropolis(initial_theta, n_samples, target_PDF, proposal_PDF, burningInFraction, lagPeriod)

# OUTPUT

# plot samples
uplt.plot_hist(theta, target_PDF)
uplt.plot_mixing(theta)
uplt.plot_autocorr(theta, 400)

plt.show()

