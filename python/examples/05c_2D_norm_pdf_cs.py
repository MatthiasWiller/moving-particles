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

from ERANataf import ERANataf
from ERADist import ERADist

# INPUT

np.random.seed(1)


initial_theta   = np.array([1.5, 1.5]).reshape(-1,1)     # initial theta
n_samples       = int(1e5)       # number of samples
burnInPeriod    = 5000            # defines burn-in-period of samples
lagPeriod       = 5              # only log every n-th value
rho_k           = 0.8

#test configuration
n_samples    = 1000
burnInPeriod = 100
lagPeriod    = 2

# target pdf
M1 = list()
M1.append(ERADist('normal', 'MOM', [1.0, 1.0]))
M1.append(ERADist('normal', 'MOM', [1.0, 2.0]))
Rho1 = np.array([[1.0, 0.6],[0.6, 1.0]])
Norm1 = ERANataf(M1, Rho1)

M2 = list()
M2.append(ERADist('normal', 'MOM', [3.0, 0.7]))
M2.append(ERADist('normal', 'MOM', [1.5, 0.3]))
Rho2 = np.array([[1.0, -0.5],[-0.5, 1.0]])
Norm2 = ERANataf(M2, Rho2)

target_PDF = lambda x: Norm1.jointpdf(x) + Norm2.jointpdf(x)

# transformation to/from U-space
transform_U2X = lambda u: Norm1.U2X(u) + Norm2.U2X(u)
transform_X2U = lambda x: Norm1.X2U(x) + Norm2.X2U(x)

# transform seed from X to U
initial_theta_u = transform_X2U(initial_theta)
theta_xx = transform_U2X(initial_theta_u)

# apply mcmc-method
theta_u = cs.cond_sampling(initial_theta_u, n_samples, rho_k, burnInPeriod, lagPeriod)

# transform chain from U to X
theta_x = np.zeros(theta_u.shape)
for i in range(0, theta_u.shape[1]):
    theta_x[:,i] = transform_U2X(theta_u[:,i]).reshape(-1)


plt.figure()
plt.scatter(theta_u[0,:], theta_u[1,:])

plt.figure()
plt.scatter(theta_x[0,:], theta_x[1,:])


plt.show()

# OUTPUT
print('E[X1] =', round(theta_x[0,:].mean(), 5))
print('E[X2] =', round(theta_x[1,:].mean(), 5))

np.save('python/data/samples_2D_pdf_cs.npy', theta_x)
print("\n> File was successfully saved!")