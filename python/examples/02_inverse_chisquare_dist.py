"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.pyplot as plt

import plots.user_plot as uplt
import algorithms.metropolis as ma

# INPUT 

# target pdf
def target_PDF(x): 
    C = 1.0
    df = 5.0
    a = 4.0
    return C* x **(-df/2) * np.exp(-a/(2*x))

# proposal pdf
def proposal_PDF():
    return np.random.uniform(0, 100, 1)

initial_theta = 1.0         # initial theta
n_samples = 500           # number of samples
burningInFraction = 0.0     # defines burning-in-period of samples
lagPeriod = 1               # only log every n-th value

# apply MCMC
theta = ma.metropolis(initial_theta, n_samples, target_PDF, proposal_PDF, burningInFraction, lagPeriod)

# OUTPUT

# plot samples
uplt.plot_hist(theta, target_PDF)
uplt.plot_mixing(theta)
uplt.plot_autocorr(theta, n_samples)

plt.show()

