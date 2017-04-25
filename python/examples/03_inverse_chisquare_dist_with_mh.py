"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.pyplot as plt

import plots.user_plot as uplt
import algorithms.metropolis_hastings as mh

# INPUT 

# target pdf (same target as in 02_inverse_chisquare)
def target_PDF(x):
    C = 1.0
    df = 5.0
    a = 4.0
    return C* x **(-df/2) * np.exp(-a/(2*x))

# proposal pdf (chisquare function with df = 2 and df = 10)
def f_prop_PDF(x, param):
    df = 2.0 # degrees of freedom
    return x **(df/2-1) * np.exp(-x/2)

# sample from proposal pdf (chisquare function with df = 2 and df = 10)
def sample_prop_PDF(param):
    df = 2.0 # degrees of freedom 
    return np.random.chisquare(df,1)

initial_theta = 1.0         # initial theta
n_samples = 10000           # number of samples
burningInFraction = 0.0     # defines burning-in-period of samples
lagPeriod = 1               # only log every n-th value

# apply MCMC
theta = mh.metropolis_hastings(initial_theta, n_samples, target_PDF, sample_prop_PDF, f_prop_PDF, burningInFraction, lagPeriod)

# OUTPUT

# plot samples
uplt.hist_plot(theta, target_PDF)
uplt.n_plot(theta)
plt.show()
