"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scps

import time as timer

import plots.sus_plot as splt
import algorithms.sus as sus

print("RUN 06_sus_example.py")


# INPUT 

np.random.seed(0) # set seed for randomization

# parameters
n_samples_per_level = 5000          # number of samples per conditional level
d                   = 10            # number of dimensions
p0                  = 0.1           # Probability of each subset, chosen adaptively

# limit-state function
beta = 3.5
LSF  = lambda u: u.sum(axis=0)/np.sqrt(d) + beta  

# distributions
mu      = 0.0
sigma   = 1.0

# marginal pdf / target pdf
#marginal_PDF    = lambda x: scps.norm.pdf(x, mu, sigma)
f_marg_PDF      = lambda x: np.exp(-0.5 * x**2)/np.sqrt(2*np.pi)

# sample from marginal pdf (standard gaussian)
#sample_marg_PDF = lambda dim: scps.norm.rvs(mu, sigma, dim)
sample_marg_PDF = lambda dim: np.random.randn(dim[0], dim[1])

# proposal distribution (uniform)
#f_prop_PDF      = lambda x, param: 0.5
f_prop_PDF      = lambda x, param: np.exp(-0.5 * x**2)/np.sqrt(2*np.pi)

# sample from proposal distribution (uniform)
#sample_prop_PDF = lambda param: np.random.uniform(param-1, param+1, 1)
sample_prop_PDF = lambda param: scps.norm.rvs(mu, sigma, 1)

# apply subset-simulation
n_loops = 1
p_F_SS_array = np.zeros(n_loops)

print('\n\n> START Sampling')
startTime = timer.time()
for i in range(0, n_loops):
    p_F_SS, theta, g = sus.subsetsim(p0, n_samples_per_level, d, sample_marg_PDF, f_marg_PDF, sample_prop_PDF, f_prop_PDF, LSF)
    p_F_SS_array[i] = p_F_SS
    print("> [", i+1, "] Subset Simulation Estimator \t=", p_F_SS)

print("\n> Time needed for Sampling =", round(timer.time() - startTime, 2), "s")


# RESULTS

print("\nEND Simulation - See results:")

p_F = scps.norm.cdf(-beta)
print("> Subset Simulation Estimator mean\t=", np.mean(p_F_SS_array))
print("> Subset Coefficient of Variation\t=", np.sqrt(np.var(p_F_SS_array))/np.mean(p_F_SS_array))
print("> Analytical probability of Failure \t=", p_F)



# OUTPUT

# analytical CDF
analytical_CDF = lambda x: scps.norm.cdf(x, beta)


# plot samples
splt.plot_sus(g, p0, n_samples_per_level, p_F_SS, analytical_CDF)
#uplt.plot_mixing(theta)
plt.show()
