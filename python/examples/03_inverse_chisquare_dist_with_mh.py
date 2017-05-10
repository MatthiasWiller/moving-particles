"""
Author: Matthias Willer 2017
"""

import numpy as np
import scipy.stats as scp
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
    #return x **(df/2-1) * np.exp(-x/2)
    return scp.chi2.pdf(x, df)

# sample from proposal pdf (chisquare function with df = 2 and df = 10)
def sample_prop_PDF(param):
    df = 2.0 # degrees of freedom 
    return scp.chi2.rvs(df,1)

initial_theta = 1.0         # initial theta
n_samples = 1000           # number of samples
burnInFraction = 0.0     # defines burn-in-period of samples
lagPeriod = 1               # only log every n-th value

# apply MCMC
theta = mh.metropolis_hastings(initial_theta, n_samples, target_PDF, sample_prop_PDF, f_prop_PDF, burnInFraction, lagPeriod)

# OUTPUT

# plot samples
uplt.plot_hist(theta[0,:], target_PDF, 1)
uplt.plot_mixing(theta[0,:])
uplt.plot_autocorr(theta[0,:], n_samples)

plt.show()
