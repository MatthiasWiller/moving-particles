"""
# ---------------------------------------------------------------------------
# Moving Particles Example 1
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
import scipy.stats as scps

import algorithms.moving_particles as mp

import algorithms.modified_metropolis as mmh
import algorithms.cond_sampling as cs

import utilities.util as uutil

print("RUN file")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT FOR MOVING PARTICLES
# ---------------------------------------------------------------------------

# parameters
N = 100                     # number of samples
d = 2                      # number of dimensions
Nb = 20                      # burn-in
sampling_method  = 'cs'     # 'mmh' = Modified Metropolis Hastings
                            # 'cs'  = Conditional Sampling
seed_selection_strategy = 2 # 
n_simulations = 5         # number of simulations

# file-name
filename = 'python/data/mp_example_1_d' + repr(d) +'_N' + repr(N) + \
           '_Nsim' + repr(n_simulations) + '_b' + repr(Nb) + '_' + sampling_method + \
           '_sss' + repr(seed_selection_strategy)

# limit-state function
# beta = 7.0345       # for pf = 10^-12
# beta = 5.9978       # for pf = 10^-9
# beta = 4.7534       # for pf = 10^-6
# beta = 4.2649       # for pf = 10^-5
# beta = 3.7190       # for pf = 10^-4
beta = 3.0902       # for pf = 10^-3
# beta = 2.3263       # for pf = 10^-2
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

# sample from marginal pdf (gaussian)
sample_marg_PDF = lambda: np.random.randn(1)

# marginal pdf / target pdf (gaussian)
f_marg_PDF      = lambda x: np.exp(-0.5 * x**2)/np.sqrt(2*np.pi)

# append distributions to list
for i in range(0, d):
    sample_marg_PDF_list.append(sample_marg_PDF)
    f_marg_PDF_list.append(f_marg_PDF)


# ---------------------------------------------------------------------------
# MOVING PARTICLES
# ---------------------------------------------------------------------------

# initializing sampling method
if sampling_method == 'mmh':
    sampler = mmh.ModifiedMetropolisHastings(sample_marg_PDF_list, f_marg_PDF_list, 'gaussian', Nb)
elif sampling_method == 'cs':
    sampler = cs.CondSampling(sample_marg_PDF_list, 0.8, Nb)

# initialization
pf_list    = []
theta_list = []
g_list     = []


start_time = timer.time()
for sim in range(0, n_simulations):
    pf_hat, theta_temp, g_temp, acc_rate, m_array = \
        mp.mp_with_seed_selection(N, LSF, sampler, sample_marg_PDF_list, seed_selection_strategy)
    
    # save simulation in list
    pf_list.append(pf_hat)
    g_list.append(g_temp)
    theta_list.append(theta_temp)

    uutil.print_simulation_progress(sim, n_simulations, start_time)


pf_sim_array = np.asarray(pf_list)
pf_mean      = np.mean(pf_sim_array)
pf_sigma     = np.std(pf_sim_array)


# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------

print("\nRESULTS:")
print("> Probability of Failure (Moving Particels Est.)\t=", round(pf_mean, 10))
print("> Probability of Failure (Analytical) \t\t\t=", round(pf_analytical, 10))
print("> Pf mean \t=", pf_mean)
print("> Pf sigma \t=", pf_sigma)
print("> C.O.V. \t=", pf_sigma/pf_mean)
print("> m = \t\t", np.sum(m_array))

# ---------------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------------

np.save(filename + '_g_list.npy', g_list)
np.save(filename + '_theta_list.npy', theta_list)
print('\n> File was successfully saved as:', filename)
