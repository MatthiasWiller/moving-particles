"""
# ---------------------------------------------------------------------------
# Moving Particles Example 1 (study-file)
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

import numpy as np
import scipy.stats as scps

import matplotlib.pyplot as plt

import algorithms.moving_particles as mp

import algorithms.modified_metropolis as mmh
import algorithms.cond_sampling as cs


# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT FOR MOVING PARTICLES
# ---------------------------------------------------------------------------

# parameters
N = 100                     # number of samples
d = 10                      # number of dimensions
Nb = 5                  # burn-in
sampling_method  = 'mmh'     # 'mmh' = Modified Metropolis Hastings
                            # 'cs'  = Conditional Sampling
n_simulations = 100          # number of simulations
seed_selection_strategy = 2

# sigma_p_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, \
#                 1.2, 1.4, 1.6, 1.8, 2.0, \
#                 2.2, 2.4, 2.6, 2.8, 3.0, \
#                 3.2, 3.4, 3.6, 3.8, 4.0, \
#                 4.2, 4.4, 4.6, 4.8, 5.0, \
#                 5.2, 5.4, 5.6, 5.8, 6.0]

sigma_p_list = [5.2, 5.4, 5.6, 5.8, 6.0]

iii = 25
for sigma_p in sigma_p_list:

    iii = iii+1
    # file-name
    filename = 'python/data/sigma_p_study/mp_example_1_d' + repr(d) +'_N' + repr(N) + \
            '_Nsim' + repr(n_simulations) + '_b' + repr(Nb) + '_' + sampling_method + \
            '_sss' + repr(seed_selection_strategy) + '_sigmap' + repr(iii)

    # limit-state function
    #beta = 5.1993       # for pf = 10^-7
    # beta = 4.7534       # for pf = 10^-6
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
        sampler = mmh.ModifiedMetropolisHastings(sample_marg_PDF_list, f_marg_PDF_list, 'gaussian', sigma_p, Nb)
    elif sampling_method == 'cs':
        rho_k = np.sqrt(1 - sigma_p**2)
        sampler = cs.CondSampling(sample_marg_PDF_list, rho_k, Nb)

    # initialization
    pf_list    = []
    theta_list = []
    g_list     = []
    m_list     = []

    for sim in range(0, n_simulations):
        pf_hat, theta_temp, g_temp, acc_rate, m_array = mp.mp_with_seed_selection(N, LSF, sampler, sample_marg_PDF_list, seed_selection_strategy)
        # save simulation in list
        pf_list.append(pf_hat)
        g_list.append(g_temp)
        theta_list.append(theta_temp)
        m_list.append(m_array)


    pf_sim_array = np.asarray(pf_list)
    pf_mean      = np.mean(pf_sim_array)
    pf_sigma     = np.std(pf_sim_array)

    # uplt.plot_m_with_poisson_dist(m_list, pf_analytical)
    # uplt.plot_mp_pf(N, g_list[0], analytical_CDF)
    # plt.show()

    # ---------------------------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------------------------

    print("\nRESULTS:")
    print("> Probability of Failure (Moving Particels Est.)\t=", round(pf_mean, 8))
    print("> Probability of Failure (Analytical) \t\t\t=", round(pf_analytical, 8))
    print("> Pf mean \t=", pf_mean)
    print("> Pf sigma \t=", pf_sigma)
    print("> C.O.V. \t=", pf_sigma/pf_mean)


    # ---------------------------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------------------------

    np.save(filename + '_g_list.npy', g_list)
    # np.save(filename + '_theta_list.npy', theta_list)
    np.save(filename + '_m_list.npy', m_list)
    print('\n> File was successfully saved as:', filename)
