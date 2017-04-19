"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as npdf

import user_plot as uplt
import mcmc as mcmc

# INPUT 

# target pdf
def target_PDF(x): 
    mu = 4    # mean
    sigma = 2  # standard deviation
    result = npdf.norm.pdf(x, mu, sigma)
    return result

# proposal pdf
def proposal_PDF():
    mu = 4 #mean
    sigma = 10 # standard deviation
    result = np.random.uniform(mu-sigma, mu+sigma, 1)
    return result

initial_theta = 0.0         # initial theta
n_samples = 10000           # number of samples
burningInFraction = 0.1     # defines burning-in-period of samples
logPeriod = 10              # only log every n-th value

# apply MCMC
theta = mcmc.mcmc(initial_theta, n_samples, target_PDF, proposal_PDF, burningInFraction, logPeriod)

# OUTPUT

# plot samples
uplt.hist_plot(theta)
uplt.n_plot(theta)
plt.show()

