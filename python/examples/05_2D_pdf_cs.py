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
# Version 2017-05
# ---------------------------------------------------------------------------
# References:
# 1."Simulation and the Monte Carlo method"
#    Rubinstein and Kroese (2017)
# ---------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scps
from scipy.special import erf
from scipy.special import erfinv

import utilities.plots as uplt
import algorithms.mcmc_metropolis_hastings as mh
import algorithms.mcmc_cond_sampling as cs

# INPUT

np.random.seed(1)


# target pdf {f(x,y) = 1/20216.335877 * exp(-(x*x*y*y + x*x + y*y - 8*x - 8*y)/2)}
c = 1/20216.335877
target_PDF = lambda u: c * np.exp(-(u[0]*u[0]*u[1]*u[1] + u[0]*u[0] + u[1]*u[1] - 8*u[0] - 8*u[1])/2)


# parameters for proposal pdf
sig = 2
mu  = [0, 0]
cov = [[sig**2, 0],[0, sig**2]]

# proposal pdf
sample_prop_PDF = lambda param: scps.norm.rvs(param, sig, 2) # draw the components independently
f_prop_PDF      = lambda x, param: scps.multivariate_normal.pdf(x, param, cov) # PDF


initial_theta   = [1.5, 1.5]     # initial theta
n_samples       = int(1e5)       # number of samples
burnInPeriod    = 5000            # defines burn-in-period of samples
lagPeriod       = 5              # only log every n-th value
rho_k           = 0.8

#test configuration
n_samples    = 1000
burnInPeriod = 100
lagPeriod    = 1


# transformation to/from U-space
phi     = lambda x: scps.norm.cdf(x)
phi_inv = lambda x: scps.norm.ppf(x)


# !! -- define target CDF and inverse of target CDF here -- !!
CDF     = lambda x: (1.25331*c*2.71828**(8/(x[1]**2+1)-0.5*(x[1]-8)*x[1])*erf(0.707107*(x[0]*x[1]**2+x[0]-4)/np.sqrt(x[1]**2+1)))/(np.sqrt(x[1]**2+1))
CDF_inv = lambda x: 1.41421*erfinv((2.24331e-8*np.exp(-8/(x[1]**2+1))*((3.55671e7*c*np.exp(0.5*(x[1]-8)*x[1])*x[1]**2/np.sqrt(x[1]**2+1))+(3.55671e7*c*np.exp(0.5*(x[1]-8)*x[1]))))/x[0])/np.sqrt(x[1]**2+1)+4/(x[1]**2+1)

transform_U2X = lambda u: CDF_inv(phi(u))
transform_X2U = lambda x: phi_inv(CDF(x))

# transform seed from X to U
initial_theta_u = transform_X2U(initial_theta)

# apply mcmc-method
theta_u = cs.cond_sampling(initial_theta_u, n_samples, rho_k, burnInPeriod, lagPeriod)

# transform chain from U to X
theta_x = transform_U2X(theta_u)

# OUTPUT
print('E[X1] =', round(theta_x[0,:].mean(), 5))
print('E[X2] =', round(theta_x[1,:].mean(), 5))

np.save('python/data/samples_2D_pdf_cs.npy', theta_x)

# plot samples
#uplt.plot_hist(theta[0,:], target_PDF, 2)
#uplt.plot_scatter_with_contour(theta, target_PDF)
# uplt.plot_mixing(theta[0, :1000])
# uplt.plot_scatter_with_hist(theta[:, :5000], target_PDF)
# uplt.plot_autocorr(theta[0,:], 50, 1)
# uplt.plot_autocorr(theta[1,:], 50, 2)
# plt.show()
