"""
# ---------------------------------------------------------------------------
# Subset Simulation Method example: Example 4.2 Ref. [2]
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
# 1."MCMC algorithms for Subset Simulation"
#    Papaioannou, Betz, Zwirglmaier, Straub (2015)
# 2. Efficiency Improvement of Stochastic Simulation by Means of Subset Sampling
#    Martin Liebscher, Stephan Pannier, Jan-Uwe Sickert, Wolfgang Graf (2006)
# ---------------------------------------------------------------------------
"""

import time as timer

import numpy as np
import scipy.stats as scps


import algorithms.cond_sampling as cs
import algorithms.modified_metropolis as mmh
import algorithms.adaptive_cond_sampling as acs

import algorithms.sus as sus

import utilities.stats as ustat
import utilities.util as uutil

print("RUN file")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT FOR SUBSET SIMULATION
# ---------------------------------------------------------------------------

# parameters
n_samples_per_level = 1000          # number of samples per conditional level
p0                  = 0.1           # Probability of each subset, chosen adaptively
sampling_method     = 'acs'         # 'mmh' = Modified Metropolis Hastings
                                    # 'cs'  = Conditional Sampling
                                    # 'acs' = adaptive Conditional Sampling
n_simulations       = 1             # Number of Simulations

# file-name
filename = 'python/data/sus_liebscher_Nspl' + repr(n_samples_per_level) + '_Nsim' + repr(n_simulations) + '_' + sampling_method

# reference solution from paper
mu_pf_mcs       = 0.00405

# parameters for beta-distribution
p = 6.0
q = 6.0
beta_distr = scps.beta(p, q, loc=-2, scale=8)

# transformation to/from U-space
phi     = lambda x: scps.norm.cdf(x)
phi_inv = lambda x: scps.norm.ppf(x)

#CDF     = lambda x: scps.beta.cdf(x, p, q)
CDF     = lambda x: beta_distr.cdf(x)
#CDF_inv = lambda x: scps.beta.ppf(x, p, q)
CDF_inv = lambda x: beta_distr.ppf(x)

transform_U2X = lambda u: CDF_inv(phi(u))
transform_X2U = lambda x: phi_inv(CDF(x))

# limit-state function
z   = lambda x: 8* np.exp(-(x[0]**2 + x[1]**2)) + 2* np.exp(-((x[0]-5)**2 + (x[1]-4)**2)) + 1 + x[0]*x[1]/10
#LSF = lambda x: 7.5 - z(x)
LSF = lambda u: 7.5 - z(transform_U2X(u))


# ---------------------------------------------------------------------------
# INPUT FOR MONTE CARLO SIMULATION (LEVEL 0)
# ---------------------------------------------------------------------------

# initialization
sample_marg_PDF_list = []
f_marg_PDF_list      = []

# sample from marginal pdf (beta-distribution)
sample_marg_PDF = lambda: transform_X2U(beta_distr.rvs(1))

# marginal pdf / target pdf (beta-distribution)
f_marg_PDF      = lambda u: beta_distr.pdf(transform_U2X(u))

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
start_time = timer.time()
for sim in range(0, n_simulations):
    # perform SubSim
    p_F_SS, theta, g = sus.subsetsim(p0, n_samples_per_level, LSF, sampler)

    # transform samples back from u to x-space
    for j in range(0, len(theta)):
        theta[j] = transform_U2X(theta[j])

    # save values in lists
    p_F_SS_list.append(p_F_SS)
    theta_list.append(theta)
    g_list.append(g)
    
    uutil.print_simulation_progress(sim, n_simulations, start_time)


print("\n> Time needed for Sampling =", round(timer.time() - start_time, 2), "s")

# computing cov
print('\n> START Computing C.O.V')
startTime = timer.time()

cov_analytical = ustat.cov_analytical(theta, g, p0, n_samples_per_level, p_F_SS)

print("> Time needed for Computing C.O.V =", round(timer.time() - startTime, 2), "s")

# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------

p_F_SS_array    = np.asarray(p_F_SS_list).reshape(-1)
sigma_pf_ss     = np.std(p_F_SS_array)
mu_pf_ss        = np.mean(p_F_SS_array)

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
