"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as npdf

import plots.user_plot as uplt
import algorithms.metropolis_hastings as mh

# INPUT 

# target pdf
def target_PDF(x):
    C = 1.0
    n = 5.0
    a = 4.0
    return C* x **(-n/2) * np.exp(-a/(2*x))

# proposal pdf
def f_prop_PDF(x):
    n = 5.0
    return x ** (n/2 - 1) * np.exp(-x/2) 

# sample from proposal pdf
def sample_prop_PDF():
    result = np.random.uniform(0, 100, 1)
    return result

initial_theta = 1.0         # initial theta
n_samples = 5000           # number of samples
burningInFraction = 0.1     # defines burning-in-period of samples
logPeriod = 10               # only log every n-th value

# apply MCMC
theta = mh.metropolis_hastings(initial_theta, n_samples, target_PDF, sample_prop_PDF, f_prop_PDF, burningInFraction, logPeriod)

# OUTPUT

# plot samples
uplt.hist_plot(theta)
uplt.n_plot(theta)
plt.show()

