"""
# ---------------------------------------------------------------------------
# Metropolis algorithm 1D example: sample Gaussian from Uniform
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

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scps

import utilities.plots as uplt
import algorithms.mcmc_metropolis as ma

# INPUT 

mu      = 4
sigma   = 2

# target pdf 
target_PDF      = lambda x: scps.norm.pdf(x, mu, sigma)

# proposal pdf
proposal_PDF    = lambda: np.random.uniform(mu-10, mu+10, 1)

initial_theta   = 0.0           # initial theta
n_samples       = 1000          # number of samples
burnInFraction  = 0.1           # defines burn-in-period of samples
lagPeriod       = 10            # only log every n-th value (thinning)

# apply MCMC
theta = ma.metropolis(initial_theta, n_samples, target_PDF, proposal_PDF, burnInFraction, lagPeriod)

# OUTPUT

# plot samples
uplt.plot_hist(theta, target_PDF)
uplt.plot_mixing(theta)
uplt.plot_autocorr(theta, 400)

plt.show()
