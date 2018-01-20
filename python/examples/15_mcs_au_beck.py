"""
# ---------------------------------------------------------------------------
# File for performing MCS on Example 5 (SDOF)
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-10
# ---------------------------------------------------------------------------
"""

import time as timer
import numpy as np

import algorithms.mcs as mcs

import SDOF as sdof


print("RUN file")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# DEFINITION OF THE WHITE NOISE ESCITATION
# ---------------------------------------------------------------------------

S  = 1                          # White noise spectral intensity 
T  = 30                         # Duration of the excitation, s
dt = 0.02                       # Time increment, s
t  = np.arange(0,T+2*dt,dt)     # time instants (one more due to interpolation)
n  = len(t)-1                   # n points ~ number of random variables
# The uncertain state vector theta consists of the sequence of i.i.d.
# standard Gaussian random variables which generate the white noise input
# at the discrete time instants
W = lambda theta: np.sqrt(2*np.pi*S/dt)*theta   # random excitation


# ---------------------------------------------------------------------------
# STANDARD INPUT
# ---------------------------------------------------------------------------

# parameters
N = int(2*1e2)       # number of Simulations

filename = 'python/data/mcs_au_beck_N' + repr(N)

# limit-state function
max_thresh = 2.4    # See Fig.(1) Ref.(1)
lsf = lambda theta: sdof.LSF(theta, t, W, max_thresh)

# ---------------------------------------------------------------------------
# INPUT FOR MONTE CARLO SIMULATION
# ---------------------------------------------------------------------------

# initialization
sample_marg_PDF_list = []

# sample from marginal pdf (gaussian)
sample_marg_PDF = lambda: np.random.randn(1)

# append distributions to list
for i in range(0, n+1):
    sample_marg_PDF_list.append(sample_marg_PDF)


# ---------------------------------------------------------------------------
# MONTE CARLO SIMULATION
# ---------------------------------------------------------------------------

print('\n> START Monte Carlo Simulation')
startTime = timer.time()

pf_mcs, theta_list, g_list = mcs.mcs(N, sample_marg_PDF_list, lsf)

# cov_mcs = np.sqrt((1 - pf_mcs) / (N * pf_mcs))

print('> Estimated Pf \t=', pf_mcs)
# print('> Estimated C.O.V. = ', cov_mcs)
print("\n> Time needed for Monte Carlo Simulation =", round(timer.time() - startTime, 2), "s")

# ---------------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------------

np.save(filename + '_g_list.npy', g_list)
np.save(filename + '_theta_list.npy', theta_list)
print("\n> File was successfully saved as:", filename)
