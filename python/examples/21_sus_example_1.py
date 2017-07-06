"""
# ---------------------------------------------------------------------------
# Subset Simulation Method example: Example 1 Ref. [1]
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
# 1."MCMC algorithms for Subset Simulation"
#    Papaioannou, Betz, Zwirglmaier, Straub (2015)
# ---------------------------------------------------------------------------
"""

import time as timer

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scps

import algorithms.sus as sus
import algorithms.cond_sampling as cs
import algorithms.modified_metropolis as mmh
import algorithms.adaptive_cond_sampling as acs

import utilities.stats as ustat
import utilities.util as uutil

print("RUN 21_sus_example_1.py")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT FOR SUBSET SIMULATION
# ---------------------------------------------------------------------------

# parameters
n_samples_per_level = 1000          # number of samples per conditional level
d                   = 10            # number of dimensions
p0                  = 0.1           # Probability of each subset, chosen adaptively
sampling_method     = 'acs'         # 'mmh' = Modified Metropolis Hastings
                                    # 'cs'  = Conditional Sampling
                                    # 'acs' = adaptive Conditional Sampling
n_simulations       = 2             # Number of Simulations

# file-name
filename = 'python/data/sus_example_1_d' + repr(d) +'_Nspl' + repr(n_samples_per_level) + '_Nsim' + repr(n_simulations) + '_' + sampling_method

# limit-state function
#beta = 5.1993       # for pf = 10^-7
#beta = 4.7534       # for pf = 10^-6
#beta = 4.2649       # for pf = 10^-5
#beta = 3.7190       # for pf = 10^-4
beta = 3.0902       # for pf = 10^-3
#beta = 2.3263       # for pf = 10^-2
LSF  = lambda u: u.sum(axis=0)/np.sqrt(d) + beta

# analytical CDF
analytical_CDF = lambda x: scps.norm.cdf(x, beta)

pf_analytical    = analytical_CDF(0)

# ---------------------------------------------------------------------------
# INPUT FOR MONTE CARLO SIMULATION (LEVEL 0)
# ---------------------------------------------------------------------------

# initialization
sample_marg_PDF_list = []
f_marg_PDF_list      = []

# sample from marginal pdf (standard gaussian)
sample_marg_PDF = lambda: np.random.randn(1)

# marginal pdf / target pdf (standard gaussian)
f_marg_PDF      = lambda x: np.exp(-0.5 * x**2)/np.sqrt(2*np.pi)

# append distributions to list
for i in range(0, d):
    sample_marg_PDF_list.append(sample_marg_PDF)
    f_marg_PDF_list.append(f_marg_PDF)


# ---------------------------------------------------------------------------
# SUBSET SIMULATION
# ---------------------------------------------------------------------------

# initializing sampling method
if sampling_method == 'mmh':
     sampler = mmh.ModifiedMetropolisHastings(sample_marg_PDF_list, f_marg_PDF_list, 'gaussian')
elif sampling_method == 'cs':
    sampler = cs.CondSampling(sample_marg_PDF_list, 0.8)
elif sampling_method == 'acs':
    sampler = acs.AdaptiveCondSampling(sample_marg_PDF_list, 0.1)


## apply subset-simulation

# initialization of lists
p_F_SS_list = []
theta_list  = []
g_list      = []

print('\n> START Sampling')
startTime = timer.time()

for i in range(0, n_simulations):
    # perform SubSim
    p_F_SS, theta, g = sus.subsetsim(p0, n_samples_per_level, LSF, sampler)

    # save values in lists
    p_F_SS_list.append(p_F_SS)
    theta_list.append(theta)
    g_list.append(g)
    print("> [", i+1, "] Subset Simulation Estimator \t=", p_F_SS)


print("\n> Time needed for Sampling =", round(timer.time() - startTime, 2), "s")

# computing cov
print('\n> START Computing C.O.V')
startTime = timer.time()

cov_analytical = ustat.cov_analytical(theta, g, p0, n_samples_per_level, p_F_SS)

print("> Time needed for Computing C.O.V =", round(timer.time() - startTime, 2), "s")


# ---------------------------------------------------------------------------
# RESULTS
# --------------------------------------------------------------------------

p_F_SS_array   = np.asarray(p_F_SS_list).reshape(-1)
sigma_pf_ss    = np.std(p_F_SS_array)
mu_pf_ss       = np.mean(p_F_SS_array)

cov_estimation = sigma_pf_ss/mu_pf_ss

print("\nRESULTS:")
print("> Probability of Failure (SubSim Est.)\t=", round(mu_pf_ss, 8))
print("> Probability of Failure (Analytical) \t=", round(pf_analytical, 8))
print("> Coefficient of Variation (Estimation)\t=", round(cov_estimation, 8))
print("> Coefficient of Variation (Analytical)\t=", round(cov_analytical, 8))


# ---------------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------------

np.save(filename + '_g_list.npy', g_list)
np.save(filename + '_theta_list.npy', theta_list)
print("\n> File was successfully saved as:", filename)
