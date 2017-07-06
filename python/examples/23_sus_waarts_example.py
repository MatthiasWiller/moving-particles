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
import matplotlib.pyplot as plt

import algorithms.modified_metropolis as mmh
import algorithms.cond_sampling as cs
import algorithms.adaptive_cond_sampling as acs

import algorithms.sus as sus

import utilities.plots as uplt
import utilities.stats as ustat
import utilities.util as uutil

print("RUN 24_sus_waarts_example.py")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT FOR SUBSET SIMULATION
# ---------------------------------------------------------------------------

# parameters
n_samples_per_level = 500          # number of samples per conditional level
p0                  = 0.1          # Probability of each subset, chosen adaptively

n_simulations = 5

# limit-state function
LSF = lambda u: np.minimum(3 + 0.1*(u[0] - u[1])**2 - 2**(-0.5) * np.absolute(u[0] + u[1]), 7* 2**(-0.5) - np.absolute(u[0] - u[1]))


# analytical CDF
# no analytical CDF available

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
# INPUT FOR MODIFIED METROPOLIS HASTINGS
# ---------------------------------------------------------------------------

# proposal distribution
#proposal_dist = 'uniform'
proposal_dist = 'gaussian'


# ---------------------------------------------------------------------------
# INPUT FOR CONDITIONAL SAMPLING
# ---------------------------------------------------------------------------

# note: don't set it to 0.2; it is too low;
rho_k = 0.8         # ~0.7 gives kinda good results

# ---------------------------------------------------------------------------
# INPUT FOR ADAPTIVE CONDITIONAL SAMPLING
# ---------------------------------------------------------------------------

#
pa = 0.1

# ---------------------------------------------------------------------------
# SUBSET SIMULATION
# ---------------------------------------------------------------------------

# initialize sampling method
#sampling_method = mmh.ModifiedMetropolisHastings(sample_marg_PDF_list, f_marg_PDF_list, proposal_dist)
sampling_method = cs.CondSampling(sample_marg_PDF_list, rho_k)
#sampling_method = acs.AdaptiveCondSampling(sample_marg_PDF_list, pa)


# apply subset-simulation

# initialization of lists
p_F_SS_list  = []
theta_list   = []
g_list       = []


print('\n> START Sampling')
startTime = timer.time()


for i in range(0, n_simulations):
    # perform SubSim
    p_F_SS, theta, g = sus.subsetsim(p0, n_samples_per_level, LSF, sampling_method)

    # save values in lists
    p_F_SS_list.append(p_F_SS)
    theta_list.append(theta)
    g_list.append(g)
    print("> [", i+1, "] Subset Simulation Estimator \t=", p_F_SS)


print("\n> Time needed for Sampling =", round(timer.time() - startTime, 2), "s")

# computing cov
print('\n> START Computing C.O.V')
startTime = timer.time()
delta     = ustat.cov_analytical(theta, g, p0, n_samples_per_level, p_F_SS)
print("> Time needed for Computing C.O.V =", round(timer.time() - startTime, 2), "s")

# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------

p_F_SS_array    = np.asarray(p_F_SS_list).reshape(-1)
sigma_pf_ss     = np.std(p_F_SS_array)
mu_pf_ss        = np.mean(p_F_SS_array)

mu_pf_mcs       = 2.275 * 10**-3

delta_analytical = delta
delta_estimation = sigma_pf_ss/mu_pf_ss

print("\nSTART Results:")
print("> Probability of Failure (SubSim Est.)\t=", round(mu_pf_ss, 8))
print("> Probability of Failure (MCS) \t\t=", round(mu_pf_mcs, 8))
print("> Coefficient of Variation (Estimation)\t=", round(delta_estimation, 8))
print("> Coefficient of Variation (Analytical)\t=", round(delta_analytical, 8))


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

# plot samples
#uplt.plot_sus_list(g_list, p0, n_samples_per_level, p_F_SS_array, analytical_CDF=0)

plt.show()
