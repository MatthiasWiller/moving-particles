"""
# ---------------------------------------------------------------------------
# Example 1 Ref. [1]: Subset Simulation vs. Moving Particles
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
# ---------------------------------------------------------------------------
"""

import numpy as np
import scipy.stats as scps

import sus as sus
import moving_particles as mp

import cond_sampling as cs
import modified_metropolis as mmh
import adaptive_cond_sampling as acs

print("RUN file")

# set seed for randomization
# np.random.seed(0)

# ---------------------------------------------------------------------------
# INPUT FOR LIMIT STATE FUNCTION
# ---------------------------------------------------------------------------
d = 10    # number of dimensions

# beta = 7.0345       # for pf = 10^-12
# beta = 5.9978       # for pf = 10^-9
beta = 4.7534       # for pf = 10^-6
# beta = 3.0902       # for pf = 10^-3

LSF  = lambda u: u.sum(axis=0)/np.sqrt(d) + beta

# ---------------------------------------------------------------------------
# INPUT FOR SUBSET SIMULATION
# ---------------------------------------------------------------------------

N_sus               = 1000  # number of samples per conditional level
p0                  = 0.1   # Probability of each subset, chosen adaptively
sampling_method_sus = 'cs'  # 'mmh' = Modified Metropolis Hastings
                            # 'cs'  = Conditional Sampling
                            # 'acs' = adaptive Conditional Sampling

# ---------------------------------------------------------------------------
# INPUT FOR MOVING PARTICLES
# ---------------------------------------------------------------------------

N_mp                 = 100  # number of initial samples samples
Nb                   = 20   # Burn-in period
sampling_method_mp   = 'cs' # 'mmh' = Modified Metropolis Hastings
                            # 'cs'  = Conditional Sampling
                            # 'acs' = adaptive Conditional Sampling
seed_selection_strat = 2    # Seed Selection Strategy

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
# RUN SUBSET SIMULATION
# ---------------------------------------------------------------------------

# initializing sampling method
if sampling_method_sus == 'mmh':
     sampler = mmh.ModifiedMetropolisHastings(sample_marg_PDF_list, f_marg_PDF_list, 'gaussian')
elif sampling_method_sus == 'cs':
    sampler = cs.CondSampling(sample_marg_PDF_list, 0.8)
elif sampling_method_sus == 'acs':
    sampler = acs.AdaptiveCondSampling(sample_marg_PDF_list, 0.1)

# perform Subset Simulation
pf_sus, theta, g = \
    sus.subsetsim(p0, N_sus, LSF, sampler)


# ---------------------------------------------------------------------------
# RUN PARTICLES
# ---------------------------------------------------------------------------

# initializing sampling method
if sampling_method_mp == 'mmh':
    sampler = mmh.ModifiedMetropolisHastings(sample_marg_PDF_list, f_marg_PDF_list, 'gaussian', 1.0, Nb)
elif sampling_method_mp == 'cs':
    sampler = cs.CondSampling(sample_marg_PDF_list, 0.8, Nb)

# perform Moving Particles
pf_mp, theta, g, acc_rate, m = \
    mp.mp_with_seed_selection(N_mp, LSF, sampler, sample_marg_PDF_list, seed_selection_strat)


# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------

print("Probability of Failure:")
print("> Analytical \t\t:", round(scps.norm.cdf(0, beta),8))
print("> Subset Simulation \t:", round(pf_sus, 8))
print("> Moving Particles \t:", round(pf_mp, 8))
