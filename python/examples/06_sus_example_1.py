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

import plots.sus_plot as splt

print("RUN 06_sus_example_1.py")

# set seed for randomization
np.random.seed(3)

# ---------------------------------------------------------------------------
# STANDARD INPUT FOR SUBSET SIMULATION
# ---------------------------------------------------------------------------

# parameters
n_samples_per_level = 1000          # number of samples per conditional level
d                   = 10            # number of dimensions
p0                  = 0.1           # Probability of each subset, chosen adaptively

# limit-state function
#beta = 3.71901        # for pf = 10^-4
#beta = 3.08899        # for pf = 10^-3
beta = 2.326          # for pf = 10^-2
LSF  = lambda u: u.sum(axis=0)/np.sqrt(d) + beta  


# ---------------------------------------------------------------------------
# INPUT FOR MODIFIED METROPOLIS HASTINGS
# ---------------------------------------------------------------------------
# distributions
mu      = 0.0
sigma   = 1.0

# marginal pdf / target pdf (standard gaussian)
f_marg_PDF      = lambda x: np.exp(-0.5 * x**2)/np.sqrt(2*np.pi)

# sample from marginal pdf (standard gaussian)
sample_marg_PDF = lambda dim: np.random.randn(dim[0], dim[1])

# proposal distribution (uniform)
#f_prop_PDF      = lambda x, param: 0.5
#f_prop_PDF      = lambda x, param: np.exp(-0.5 * x**2)/np.sqrt(2*np.pi)
f_prop_PDF      = lambda x, param: ( 2.*np.pi*sigma**2. )**-.5 * np.exp( -.5 * (x-param)**2. / sigma**2. )

# sample from proposal distribution (uniform)
#sample_prop_PDF = lambda param: np.random.uniform(param-1, param+1, 1)
#sample_prop_PDF = lambda param: scps.norm.rvs(mu, sigma, 1)
sample_prop_PDF = lambda param: np.random.normal(param, sigma, 1)


# ---------------------------------------------------------------------------
# INPUT FOR CONDITIONAL SAMPLING
# ---------------------------------------------------------------------------

# sample from marginal pdf (standard gaussian)
sample_marg_PDF = lambda dim: np.random.randn(dim[0], dim[1])

# sample from conditional PDF
sample_cond_PDF = lambda mu_cond, sigma_cond: np.random.normal(mu_cond, sigma_cond, 1)

# note: don't set it to 0.2; it is too low; 
# 0.7 gives kinda good results
rho_k = 0.8

# ---------------------------------------------------------------------------
# INPUT FOR ADAPTIVE CONDITIONAL SAMPLING
# ---------------------------------------------------------------------------

# sample from marginal pdf (standard gaussian)
sample_marg_PDF = lambda dim: np.random.randn(dim[0], dim[1])

# sample from conditional PDF
sample_cond_PDF = lambda mu_cond, sigma_cond: np.random.normal(mu_cond, sigma_cond, 1)

pa = 0.1


# ---------------------------------------------------------------------------
# SUBSET SIMULATION
# ---------------------------------------------------------------------------

# initializing sampling method
#sampling_method = mmh.ModifiedMetropolisHastings(sample_marg_PDF, f_marg_PDF, sample_prop_PDF, f_prop_PDF)
#sampling_method = cs.CondSampling(sample_marg_PDF, sample_cond_PDF, rho_k)
sampling_method = acs.AdaptiveCondSampling(sample_marg_PDF, sample_cond_PDF, pa)

# apply subset-simulation
n_loops      = 10
#p_F_SS_array = np.zeros(n_loops)

p_F_SS_list  = []
theta_list   = []
g_list       = []

print('\n> START Sampling')
startTime = timer.time()
for i in range(0, n_loops):
    # perform SubSim
    p_F_SS, theta, g = sus.subsetsim(p0, n_samples_per_level, d, LSF, sampling_method)
    #p_F_SS_array = p_F_SS
    # save values in lists
    p_F_SS_list.append(p_F_SS)
    theta_list.append(theta)
    g_list.append(g)
    print("> [", i+1, "] Subset Simulation Estimator \t=", p_F_SS)

print("\n> Time needed for Sampling =", round(timer.time() - startTime, 2), "s")

# computing cov
print('\n> START Computing C.O.V')
startTime = timer.time()
delta = sus.cov_analytical(theta, g, p0, n_samples_per_level, p_F_SS)
print("> Time needed for Computing C.O.V =", round(timer.time() - startTime, 2), "s")

# ---------------------------------------------------------------------------
# RESULTS
# --------------------------------------------------------------------------

p_F_SS_array = np.asarray(p_F_SS_list).reshape(-1)
sigma_pf_ss = np.std(p_F_SS_array)
mu_pf_ss = np.mean(p_F_SS_array)

print("\nRESULTS:")
print("> Probability of Failure (SubSim Est.)\t=", np.mean(p_F_SS_array))
print("> Probability of Failure (Analytical) \t=", round(scps.norm.cdf(-beta), 8))
print("> Coefficient of Variation (Estimation)\t=", sigma_pf_ss/mu_pf_ss)
print("> Coefficient of Variation (Analytical)\t=", round(delta, 8))



# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

# analytical CDF
analytical_CDF = lambda x: scps.norm.cdf(x, beta)

# plot samples
#splt.plot_sus(g, p0, n_samples_per_level, p_F_SS, analytical_CDF)
splt.plot_sus_list(g_list, p0, n_samples_per_level, p_F_SS_array, analytical_CDF)
plt.show()
