"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scps

import plots.user_plot as uplt
import algorithms.sus as sus

# INPUT 


np.random.seed(1)

#burnInFraction = 0.0     # defines burn-in-period of samples
#lagPeriod = 1               # only log every n-th value

# parameters
n_samples_per_level = 100         # number of samples per conditional level
d  = 10      # number of dimensions
p0 = 0.1     # Probability of each subset, chosen adaptively

# limit-state function
beta = 3.5
LSF    = lambda u: -u.sum(axis=0)/np.sqrt(d) + beta  

# target pdf (two weighted gaussian)
def marginal_PDF(x):
    mu = 0
    sigma = 1
    return scps.norm.pdf(x, mu, sigma)

# proposal pdf 
def sample_prop_PDF(param):
    mu = 0
    sigma = 1
    return scps.norm.rvs(mu, sigma, 1)

# proposal pdf
def f_prop_PDF(x, param):
    mu = 0
    sigma = 1
    return scps.norm.pdf(x, mu, sigma)

# apply subset-simulation
#theta = mh.metropolis_hastings(initial_theta, n_samples, target_PDF, sample_prop_PDF, f_prop_PDF, burnInFraction, lagPeriod)

p_F_SS, theta = sus.subsetsim(p0, n_samples_per_level, d, marginal_PDF, sample_prop_PDF, f_prop_PDF, LSF)


print("finished simulation")

p_F = scps.norm.cdf(-beta)
print("> Subset Simulation Estimator =", p_F_SS)
print("> Analytical probability of Failure =", p_F)
# OUTPUT

# plot samples
#uplt.plot_hist(theta, target_PDF)
#uplt.plot_mixing(theta)
#plt.show()
