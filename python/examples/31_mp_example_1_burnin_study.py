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
# Version 2017-07
# ---------------------------------------------------------------------------
"""

import numpy as np
import scipy.stats as scps

import matplotlib.pyplot as plt

import algorithms.moving_particles as mp

import algorithms.modified_metropolis as mmh
import algorithms.cond_sampling as cs

import utilities.plots as uplt

print("RUN 31_mp_example_1.py")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT FOR MOVING PARTICLES
# ---------------------------------------------------------------------------

# parameters
N = 10                     # number of samples
d = 10                      # number of dimensions
#b_max = 11                  # burn-in
sampling_method  = 'cs'     # 'mmh' = Modified Metropolis Hastings
                            # 'cs'  = Conditional Sampling
n_simulations = 50          # number of simulations

#burn_in_list = [i for i in range(1, b_max)]
burn_in_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
seed_selection_list = [0, 1, 2, 3, 4, 5]
# burn_in_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# seed_selection_list = [2, 3, 4, 5]


for seed_selection_strategy in seed_selection_list:
    for burnin in burn_in_list:

        # file-name
        filename = 'python/data/burnin_study_N10/mp_example_1_d' + repr(d) +'_N' + repr(N) + \
                '_Nsim' + repr(n_simulations) + '_b' + repr(burnin) + '_' + sampling_method + \
                '_sss' + repr(seed_selection_strategy)

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
            sampler = mmh.ModifiedMetropolisHastings(sample_marg_PDF_list, f_marg_PDF_list, 'gaussian', burnin)
        elif sampling_method == 'cs':
            sampler = cs.CondSampling(sample_marg_PDF_list, 0.8, burnin)

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
