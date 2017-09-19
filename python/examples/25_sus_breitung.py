"""
# ---------------------------------------------------------------------------
# Subset Simulation Method example
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-07
# ---------------------------------------------------------------------------
# References:
# 1."MCMC algorithms for Subset Simulation"
#    Papaioannou, Betz, Zwirglmaier, Straub (2015)
# 2. Efficiency Improvement of Stochastic Simulation by Means of Subset Sampling
#    Martin Liebscher, Stephan Pannier, Jan-Uwe Sickert, Wolfgang Graf (2006)
# ---------------------------------------------------------------------------
"""

import time as timer

import numpy as np

import algorithms.modified_metropolis as mmh
import algorithms.cond_sampling as cs
import algorithms.adaptive_cond_sampling as acs

import algorithms.sus as sus

import utilities.stats as ustat

print("RUN 25_sus_breitung.py")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT FOR SUBSET SIMULATION
# ---------------------------------------------------------------------------

# parameters
n_samples_per_level = 500          # number of samples per conditional level
p0                  = 0.1          # Probability of each subset, chosen adaptively
sampling_method     = 'cs'         # 'mmh' = Modified Metropolis Hastings
                                    # 'cs'  = Conditional Sampling
                                    # 'acs' = adaptive Conditional Sampling
n_simulations       = 100             # Number of Simulations


# file-name
filename = 'python/data/sus_breitung_b_Nspl' + repr(n_samples_per_level) + '_Nsim' + repr(n_simulations) + '_' + sampling_method

# reference solution from paper
mu_pf_mcs       = 3.17 * 10**-5

# limit-state function
# LSF = lambda x: np.minimum(5-x[0], 4+x[1])
LSF = lambda x: np.minimum(5-x[0], 1/(1+np.exp(-2*(x[1]+4)))-0.5)

# ---------------------------------------------------------------------------
# INPUT FOR MONTE CARLO SIMULATION (LEVEL 0)
# ---------------------------------------------------------------------------

# initialization
sample_marg_PDF_list = []
f_marg_PDF_list      = []

# sample from marginal pdf (gaussian)
sample_marg_PDF = lambda: np.random.randn(1)

# marginal pdf / target pdf (gaussian)
f_marg_PDF      = lambda x: np.exp(-0.5 * x**2)/np.sqrt(2*np.pi)

# append distributions to list
sample_marg_PDF_list.append(sample_marg_PDF)
sample_marg_PDF_list.append(sample_marg_PDF)
f_marg_PDF_list.append(f_marg_PDF)
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
p_F_SS_list  = []
theta_list   = []
g_list       = []


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
# ---------------------------------------------------------------------------

p_F_SS_array   = np.asarray(p_F_SS_list).reshape(-1)
sigma_pf_ss    = np.std(p_F_SS_array)
mu_pf_ss       = np.mean(p_F_SS_array)

cov_estimation = sigma_pf_ss/mu_pf_ss

print("\nSTART Results:")
print("> Probability of Failure (SubSim Est.)\t=", round(mu_pf_ss, 8))
print("> Probability of Failure (MCS) \t\t=", round(mu_pf_mcs, 8))
print("> Coefficient of Variation (Estimation)\t=", round(cov_estimation, 8))
print("> Coefficient of Variation (Analytical)\t=", round(cov_analytical, 8))


# ---------------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------------

np.save(filename + '_g_list.npy', g_list)
np.save(filename + '_theta_list.npy', theta_list)
print("\n> File was successfully saved as:", filename)
