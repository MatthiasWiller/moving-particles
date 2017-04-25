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
    mu = 4    # mean
    sigma = 2  # standard deviation
    return npdf.norm.pdf(x, mu, sigma)

# proposal pdf
def proposal_PDF():
    mu = 4 #mean
    sigma = 10 # standard deviation
    return np.random.uniform(mu-sigma, mu+sigma, 1)

initial_theta = 0.0         # initial theta
n_samples = 100           # number of samples
burningInFraction = 0.1     # defines burning-in-period of samples
logPeriod = 5              # only log every n-th value

# apply MCMC
theta = ma.metropolis(initial_theta, n_samples, target_PDF, proposal_PDF, burningInFraction, logPeriod)

# r = np.correlate(theta, theta, "full")


# OUTPUT

# plot samples
uplt.hist_plot(theta, target_PDF)
uplt.n_plot(theta)
plt.show()
