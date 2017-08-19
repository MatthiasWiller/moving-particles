"""
# ---------------------------------------------------------------------------
# Subset Simulation Method example: Example 2 Ref. [1]
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

import utilities.plots as uplt
import utilities.stats as ustat
import utilities.util as uutil

print("RUN 07_sus_example_2.py")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT FOR SUBSET SIMULATION
# ---------------------------------------------------------------------------

# parameters
n_samples_per_level = 1000          # number of samples per conditional level
d                   = 100            # number of dimensions
p0                  = 0.1           # Probability of each subset, chosen adaptively

# transformation to/from U-space
phi     = lambda x: scps.norm.cdf(x)
phi_inv = lambda x: scps.norm.ppf(x)

CDF     = lambda x: (x >= 0) * (1 - np.exp(-lam * x))
CDF_inv = lambda x: (x < 1) * (x >= 0) * (-np.log(1 - x)/lam)

transform_U2X = lambda u: CDF_inv(phi(u))
transform_X2U = lambda x: phi_inv(CDF(x))

# limit-state function
Ca    = 135
LSF_A = lambda u: Ca - transform_U2X(u).sum(axis=0)
#LSF_A = lambda u: Ca - u.sum(axis=0)
Cb    = 0
LSF_B = lambda u: -Cb + transform_U2X(u).sum(axis=0)
#LSF_B = lambda u: -Cb + u.sum(axis=0)

# analytical CDF
lam = 1.0
analytical_CDF = lambda x: 1 - scps.gamma.cdf(Ca - x, d, lam)


# ---------------------------------------------------------------------------
# INPUT FOR MONTE CARLO SIMULATION (LEVEL 0)
# ---------------------------------------------------------------------------

# initialization
sample_marg_PDF_list = []
f_marg_PDF_list      = []

# sample from marginal pdf (exponential)
sample_marg_PDF = lambda: transform_X2U(np.random.exponential(1/lam, 1))
#sample_marg_PDF = lambda: np.random.exponential(1/lam, 1)

# marginal pdf / target pdf (exponential)
f_marg_PDF      = lambda u: int(u > 0) * lam * np.exp(-lam*transform_U2X(u))
#f_marg_PDF      = lambda x: int(x > 0) * lam * np.exp(-lam*x)

# append distributions to list
for i in range(0, d):
    sample_marg_PDF_list.append(sample_marg_PDF)
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

# initializing sampling method
#sampling_method = mmh.ModifiedMetropolisHastings(sample_marg_PDF_list, f_marg_PDF_list, proposal_dist)
sampling_method = cs.CondSampling(sample_marg_PDF_list, rho_k)
#sampling_method = acs.AdaptiveCondSampling(sample_marg_PDF_list, pa)


# apply subset-simulation
n_sim = 10

# initialization of lists
p_F_SS_list  = []
theta_list   = []
g_list       = []

print('\n> START Sampling')
startTime = timer.time()

n_loops = n_sim
while n_loops > 0:
    for i in range(0, n_loops):
        # perform SubSim
        p_F_SS, theta, g = sus.subsetsim(p0, n_samples_per_level, LSF_A, sampling_method)

        # save values in lists
        p_F_SS_list.append(p_F_SS)
        theta_list.append(theta)
        g_list.append(g)
        print("> [", i+1, "] Subset Simulation Estimator \t=", p_F_SS)

    n_eff_sim = uutil.get_n_eff_sim(g_list)
    n_loops = n_sim - n_eff_sim

print("\n> Time needed for Sampling =", round(timer.time() - startTime, 2), "s")

# computing cov
print('\n> START Computing C.O.V')
startTime = timer.time()
delta     = ustat.cov_analytical(theta, g, p0, n_samples_per_level, p_F_SS)
print("> Time needed for Computing C.O.V =", round(timer.time() - startTime, 2), "s")

# ---------------------------------------------------------------------------
# RESULTS
# --------------------------------------------------------------------------

p_F_SS_array    = np.asarray(p_F_SS_list).reshape(-1)
sigma_pf_ss     = np.std(p_F_SS_array)
mu_pf_ss        = np.mean(p_F_SS_array)

pf_analytical    = analytical_CDF(0)

delta_analytical = delta
delta_estimation = sigma_pf_ss/mu_pf_ss

print("\nRESULTS:")
print("> Probability of Failure (SubSim Est.)\t=", round(mu_pf_ss, 8))
print("> Probability of Failure (Analytical) \t=", round(pf_analytical, 8))
print("> Coefficient of Variation (Estimation)\t=", round(delta_estimation, 8))
print("> Coefficient of Variation (Analytical)\t=", round(delta_analytical, 8))


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

# plot samples
uplt.plot_sus_list(g_list, p0, n_samples_per_level, p_F_SS_array, analytical_CDF)
plt.show()