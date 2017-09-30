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

import scipy.integrate as integrate
import scipy.interpolate as interpolate

import utilities.plots as uplt
import algorithms.mcmc_metropolis_hastings as mh
import algorithms.mcmc_cond_sampling as cs

# INPUT

np.random.seed(1)


# target pdf 
c = 0 # correlation rho = 0
# c = 1 # correlation rho = -0.40
target_PDF = lambda x: (1 - c*(1-x[0]-x[1])+c*c*x[0]*x[1])*np.exp(-(x[0]+x[1]+c*x[0]*x[1]))


initial_theta   = np.array([1.5, 1.5]).reshape(-1,1)    # initial theta
n_samples       = int(1e4)       # number of samples
burnInPeriod    = 1000            # defines burn-in-period of samples
lagPeriod       = 5              # only log every n-th value
rho_k           = 0.8

#test configuration
# n_samples    = 1000
# burnInPeriod = 100
# lagPeriod    = 2


# transformation to/from U-space
phi     = lambda x: scps.norm.cdf(x)
phi_inv = lambda x: scps.norm.ppf(x)


# !! -- define target CDF and inverse of target CDF here -- !!
f = lambda x, y: c * np.exp(-(x*x*y*y + x*x + y*y - 8*x - 8*y)/2)

x_line = np.linspace(-5, 10, 500)
f_marg = np.zeros(len(x_line))

for i in range(0, len(x_line)):
    print('integrating pdf:',i/len(x_line)*100,'%')
    f_marg[i], err = integrate.quad(f, -10, 15, args=(x_line[i]))

# normalize
f_marg = f_marg/np.sum(f_marg)
F_marg = np.cumsum(f_marg)

# CDF     = lambda x: np.array([(1-np.exp(-x[0])), (1-(1+c*x[1])*np.exp(-(x[1]+c*x[0]*x[1])))])
# CDF_inv = lambda x: np.array([interpolate.spline(F_marg,x_line,x[0]), interpolate.spline(F_marg,x_line,x[1])])

CDF     = lambda x: np.array([1-np.exp(-x[0]), 1-np.exp(-x[1])])
CDF_inv = lambda x: np.array([-np.log(1-x[0]), -np.log(1-x[1])])


transform_U2X = lambda u: CDF_inv(phi(u))
transform_X2U = lambda x: phi_inv(CDF(x))

# transform seed from X to U
initial_theta_u = transform_X2U(initial_theta)

# apply mcmc-method
theta_u = cs.cond_sampling(initial_theta_u, n_samples, rho_k, burnInPeriod, lagPeriod)

# transform chain from U to X
theta_x = transform_U2X(theta_u)

plt.figure()
plt.scatter(theta_u[0,:], theta_u[1,:])

plt.figure()
plt.scatter(theta_x[0,:], theta_x[1,:])

plt.show()

# OUTPUT
print('E[X1] =', round(theta_x[0,:].mean(), 5))
print('E[X2] =', round(theta_x[1,:].mean(), 5))

np.save('python/data/samples_2D_exp_pdf_cs.npy', theta_x)

