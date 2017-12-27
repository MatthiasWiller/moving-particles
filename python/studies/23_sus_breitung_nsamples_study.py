"""
# ---------------------------------------------------------------------------
# Subset Simulation example 3 (study-file)
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

import algorithms.sus as sus

import algorithms.modified_metropolis as mmh
import algorithms.cond_sampling as cs
import algorithms.adaptive_cond_sampling as acs

import utilities.util as uutil

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT FOR MOVING PARTICLES
# ---------------------------------------------------------------------------

# parameters
p0 = 0.1                # Probability of each subset, chosen adaptively
sampling_method = 'cs'  # 'mmh' = Modified Metropolis Hastings
                        # 'cs'  = Conditional Sampling
                        # 'acs' = adaptive Conditional Sampling
n_simulations = 100     # number of simulations

nsamples_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]

direction = 'python/data/example3/nsamples_study_sus/'

for N in nsamples_list:

    # file-name
    filename = direction + 'sus_breitung_N' + repr(N) + \
            '_Nsim' + repr(n_simulations) + '_' + sampling_method

    # limit-state function
    # LSF = lambda x: np.minimum(5-x[0], 4+x[1])
    LSF = lambda x: np.minimum(5-x[0], 1/(1+np.exp(-2*(x[1]+4)))-0.5)

    # reference solution from paper
    pf_mcs       = 3.17e-5

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
    # MOVING PARTICLES
    # ---------------------------------------------------------------------------

    # initializing sampling method
    if sampling_method == 'mmh':
        sampler = mmh.ModifiedMetropolisHastings(sample_marg_PDF_list, f_marg_PDF_list, 'gaussian', 1.0, 0)
    elif sampling_method == 'cs':
        sampler = cs.CondSampling(sample_marg_PDF_list, 0.8, 0)
    elif sampling_method == 'acs':
        sampler = acs.AdaptiveCondSampling(sample_marg_PDF_list, 0.1)
    
    ## apply subset-simulation

    # initialization
    pf_list    = []
    theta_list = []
    g_list     = []

    start_time = timer.time()
    for sim in range(0, n_simulations):
        pf_hat, theta_temp, g_temp = \
            sus.subsetsim(p0, N, LSF, sampler)

        # save simulation in list
        pf_list.append(pf_hat)
        g_list.append(g_temp)
        theta_list.append(theta_temp)

        uutil.print_simulation_progress(sim, n_simulations, start_time)

    pf_sim_array = np.asarray(pf_list).reshape(-1)
    pf_mean      = np.mean(pf_sim_array)
    pf_sigma     = np.std(pf_sim_array)

    # ---------------------------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------------------------

    print("\nRESULTS:")
    print("> Probability of Failure (Moving Particels Est.)\t=", round(pf_mean, 8))
    print("> Probability of Failure (Monte Carlo Simulation) \t=", round(pf_mcs, 8))
    print("> Pf mean \t=", pf_mean)
    print("> Pf sigma \t=", pf_sigma)
    print("> C.O.V. \t=", pf_sigma/pf_mean)


    # ---------------------------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------------------------

    np.save(filename + '_g_list.npy', g_list)
    # np.save(filename + '_theta_list.npy', theta_list)
    print('\n> File was successfully saved as:', filename)
