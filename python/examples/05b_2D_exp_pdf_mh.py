"""
# ---------------------------------------------------------------------------
# Metropolis-Hastings algorithm 2D example: Example 6.4 Ref. [1]
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-10
# ---------------------------------------------------------------------------
# References:
# 1."Simulation and the Monte Carlo method"
#    Rubinstein and Kroese (2017)
# ---------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scps

import utilities.plots as uplt
import algorithms.mcmc_metropolis_hastings as mh

# INPUT

np.random.seed(1)


# target pdf 
c = 0 # correlation rho = 0
# c = 1 # correlation rho = -0.40
target_PDF = lambda x: (1 - c*(1-x[0]-x[1])+c*c*x[0]*x[1])*np.exp(-(x[0]+x[1]+c*x[0]*x[1]))


# parameters for proposal pdf
sig = 2
mu  = [0, 0]
cov = [[sig**2, 0],[0, sig**2]]

# proposal pdf
# sample_prop_PDF = lambda param: scps.norm.rvs(mu, sig, 2) # draw the components independently
# f_prop_PDF      = lambda x, param: scps.multivariate_normal.pdf(x, mu, cov) # PDF
sample_prop_PDF = lambda param: scps.norm.rvs(param, sig, 2) # draw the components independently
f_prop_PDF      = lambda x, param: scps.multivariate_normal.pdf(x, param, cov) # PDF


initial_theta   = np.array([1.5, 1.5]).reshape(-1,1)    # initial theta
n_samples       = int(1e4)       # number of samples
burnInPeriod    = 1000            # defines burn-in-period of samples
lagPeriod       = 5              # only log every n-th value

#test configuration
# n_samples = 1000
# burnInPeriod   = 100
# lagPeriod      = 1


# apply mcmc-method
theta = mh.metropolis_hastings(initial_theta, n_samples, target_PDF, sample_prop_PDF, f_prop_PDF, burnInPeriod, lagPeriod)

plt.figure()
plt.scatter(theta[0,:], theta[1,:])
plt.show()

# OUTPUT
print('E[X1] =', round(theta[0,:].mean(), 5))
print('E[X2] =', round(theta[1,:].mean(), 5))

np.save('python/data/samples_2D_exp_pdf_mh.npy', theta)
print("\n> File was successfully saved!")
