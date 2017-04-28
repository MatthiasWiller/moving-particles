"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as npdf

import plots.user_plot as uplt
import algorithms.metropolis_hastings as mh

# INPUT 

# target pdf (two weighted gaussian)
def target_PDF(x):
    mu = [0.0, 10.0] 
    sigma = [2.0, 2.0]
    w = [0.3, 0.7]
    return w[0]*npdf.norm.pdf(x, mu[0], sigma[0]) + w[1]*npdf.norm.pdf(x, mu[1], sigma[1])

# proposal pdf 
def sample_prop_PDF(mu):
    sigma = 10
    return np.random.normal(mu, sigma, 1)

# proposal pdf
def f_prop_PDF(x, param):
    mu = param
    sigma = 10
    return npdf.norm.pdf(x, mu, sigma)

np.random.seed(1)
initial_theta = 20*np.random.uniform(-1.0, 1.0, 1)         # initial theta
n_samples = 400           # number of samples
burnInFraction = 0.2     # defines burn-in-period of samples
lagPeriod = 1               # only log every n-th value

# apply MCMC
theta = mh.metropolis_hastings(initial_theta, n_samples, target_PDF, sample_prop_PDF, f_prop_PDF, burnInFraction, lagPeriod)

# OUTPUT

# plot samples
uplt.hist_plot(theta, target_PDF)
uplt.n_plot(theta)
uplt.estimated_autocorrelation(theta[0, :], 1000)

plt.show()
