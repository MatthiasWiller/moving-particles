"""
# ---------------------------------------------------------------------------
# File for performing MCS
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-07
# ---------------------------------------------------------------------------
"""

import time as timer

import numpy as np

import algorithms.mcs as mcs


print("RUN 15_mcs_breitung.py")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT
# ---------------------------------------------------------------------------

# parameters
N = int(1e7)       # number of Simulations

filename = 'python/data/mcs_breitung_b_N' + repr(N)

# limit-state function
# LSF = lambda x: np.minimum(5-x[0], 4+x[1])
LSF = lambda x: np.minimum(5-x[0], 1/(1+np.exp(-2*(x[1]+4)))-0.5)
# ---------------------------------------------------------------------------
# INPUT FOR MONTE CARLO SIMULATION
# ---------------------------------------------------------------------------

# initialization
sample_marg_PDF_list = []

# sample from marginal pdf (gaussian)
sample_marg_PDF = lambda: np.random.randn(1)

# append distributions to list
sample_marg_PDF_list.append(sample_marg_PDF)
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
