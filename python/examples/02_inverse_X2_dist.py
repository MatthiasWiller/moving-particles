"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as npdf

import plots.user_plot as uplt
import algorithms.metropolis as ma

# INPUT 

# target pdf
def target_PDF(x): 
    C = 1.0
    n = 5.0
    a = 4.0
    return C* x **(-n/2) * np.exp(-a/(2*x))

# proposal pdf
def proposal_PDF():
    return np.random.uniform(0, 100, 1)

initial_theta = 1.0         # initial theta
n_samples = 5000           # number of samples
burningInFraction = 0.1     # defines burning-in-period of samples
logPeriod = 10               # only log every n-th value

# apply MCMC
theta = ma.metropolis(initial_theta, n_samples, target_PDF, proposal_PDF, burningInFraction, logPeriod)

# OUTPUT

# plot samples
uplt.hist_plot(theta)
uplt.n_plot(theta)
plt.show()
