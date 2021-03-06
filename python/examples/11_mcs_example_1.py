"""
# ---------------------------------------------------------------------------
# File for performing MCS for Example 1
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

print("RUN file")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT
# ---------------------------------------------------------------------------

# parameters
N = int(1e6)       # number of Simulations
d = 10

filename = 'python/data/mcs_example_1_d' + repr(d) +'_N' + repr(N)

# limit-state function
#beta = 5.1993       # for pf = 10^-7
#beta = 4.7534       # for pf = 10^-6 (N=5*1e9)
#beta = 4.2649       # for pf = 10^-5 (N=5*1e8)
#beta = 3.7190       # for pf = 10^-4 (N=5*1e7)
beta = 3.0902       # for pf = 10^-3 (N=5*1e6)
#beta = 2.3263       # for pf = 10^-2 (N=5*1e5)
LSF  = lambda u: u.sum(axis=0)/np.sqrt(d) + beta

# ---------------------------------------------------------------------------
# INPUT FOR MONTE CARLO SIMULATION
# ---------------------------------------------------------------------------

# initialization
sample_marg_PDF_list = []

# sample from marginal pdf (gaussian)
sample_marg_PDF = lambda: np.random.randn(1)

# append distributions to list
for i in range(0, d):
    sample_marg_PDF_list.append(sample_marg_PDF)


# ---------------------------------------------------------------------------
# MONTE CARLO SIMULATION
# ---------------------------------------------------------------------------

print('\n> START Monte Carlo Simulation')
startTime = timer.time()

pf_mcs, theta_list, g_list = mcs.mcs(N, sample_marg_PDF_list, LSF)

cov_mcs = np.sqrt((1 - pf_mcs) / (N * pf_mcs))

print('> Estimated Pf \t=', pf_mcs)
print('> Estimated C.O.V. = ', cov_mcs)
print("\n> Time needed for Monte Carlo Simulation =", round(timer.time() - startTime, 2), "s")

# ---------------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------------

np.save(filename + '_g_list.npy', g_list)
np.save(filename + '_theta_list.npy', theta_list)
print("\n> File was successfully saved as:", filename)
