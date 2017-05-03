"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as npdf

import plots.user_plot as uplt
import algorithms.sus as sus

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

# limit state function g(x) (=LSF) failure at g(x) > b
def LSF(x):
    return np.sum(x)

np.random.seed(1)
initial_theta = 20*np.random.uniform(-1.0, 1.0, 1)         # initial theta

burnInFraction = 0.0     # defines burn-in-period of samples
lagPeriod = 1               # only log every n-th value

p0 = 0.1
n_samples = 100         # number of samples per conditional level

# apply subset-simulation
#theta = mh.metropolis_hastings(initial_theta, n_samples, target_PDF, sample_prop_PDF, f_prop_PDF, burnInFraction, lagPeriod)

p_F = sus.subsetsim(p0,n_samples,target_PDF,proposal_PDF, LSF)

# OUTPUT

# plot samples
uplt.hist_plot(theta, target_PDF)
uplt.n_plot(theta)
plt.show()
