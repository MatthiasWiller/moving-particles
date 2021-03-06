"""
# ---------------------------------------------------------------------------
# Moving Particles Example 4 (study-file)
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

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT FOR MOVING PARTICLES
# ---------------------------------------------------------------------------

# parameters
N = 100                     # number of samples
Nb = 5                      # burn-in
sampling_method  = 'mmh'     # 'mmh' = Modified Metropolis Hastings
                            # 'cs'  = Conditional Sampling
n_simulations = 100          # number of simulations
seed_selection_strategy = 2

# sigma_p_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
sigma_p_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, \
                1.2, 1.4, 1.6, 1.8, 2.0, \
                2.2, 2.4, 2.6, 2.8, 3.0, \
                3.2, 3.4, 3.6, 3.8, 4.0, \
                4.2, 4.4, 4.6, 4.8, 5.0]

direction = 'python/data/example4/sigma_p_study_mp/'

iii = 0
for sigma_p in sigma_p_list:

    iii = iii+1
    # file-name
    filename = direction + 'mp_liebscher_N' + repr(N) + \
            '_Nsim' + repr(n_simulations) + '_b' + repr(Nb) + '_' + sampling_method + \
            '_sss' + repr(seed_selection_strategy) + '_sigmap' + repr(iii)

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

    # reference solution from paper
    pf_mcs       = 4.05e-3

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
        sampler = mmh.ModifiedMetropolisHastings(sample_marg_PDF_list, f_marg_PDF_list, 'gaussian', sigma_p, Nb)
    elif sampling_method == 'cs':
        rho_k = np.sqrt(1 - sigma_p**2)
        sampler = cs.CondSampling(sample_marg_PDF_list, rho_k, Nb)

    # initialization
    pf_list    = []
    theta_list = []
    g_list     = []
    m_list     = []

    start_time = timer.time()
    for sim in range(0, n_simulations):
        pf_hat, theta_temp, g_temp, acc_rate, m_array = \
            mp.mp_with_seed_selection(N, LSF, sampler, sample_marg_PDF_list, seed_selection_strategy)
        
        # transform samples back from u to x-space
        # for j in range(0, len(theta_temp)):
        #     theta_temp[j] = transform_U2X(theta_temp[j])

        # save simulation in list
        pf_list.append(pf_hat)
        g_list.append(g_temp)
        theta_list.append(theta_temp)
        m_list.append(m_array)

        uutil.print_simulation_progress(sim, n_simulations, start_time)


    pf_sim_array = np.asarray(pf_list)
    pf_mean      = np.mean(pf_sim_array)
    pf_sigma     = np.std(pf_sim_array)


    # ---------------------------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------------------------

    print("\nRESULTS:")
    print("> Probability of Failure (Moving Particels Est.)\t=", round(pf_mean, 8))
    print("> Probability of Failure (Monte Carlo Simulation)\t=", round(pf_mcs, 8))
    print("> Pf mean \t=", pf_mean)
    print("> Pf sigma \t=", pf_sigma)
    print("> C.O.V. \t=", pf_sigma/pf_mean)


    # ---------------------------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------------------------

    np.save(filename + '_g_list.npy', g_list)
    # np.save(filename + '_theta_list.npy', theta_list)
    # np.save(filename + '_m_list.npy', m_list)
    print('\n> File was successfully saved as:', filename)
