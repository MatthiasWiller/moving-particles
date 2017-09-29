"""
# ---------------------------------------------------------------------------
# Moving Particles Method
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-07
# ---------------------------------------------------------------------------
# Input:
# * theta0          : seed of the Markov-chain
# * N               : number of samples of Markov-chain (including seed)
# * f_marg_PDF      : marginal pdf
# * sample_prop_PDF : function to sample from proposal pdf
# * f_prop_PDF      : proposal pdf
# * LSF             : limit state function
# * b               : threshold level of the limit state function
# ---------------------------------------------------------------------------
# Output:
# * theta   : samples of Markov-chain generated from the seed (theta0)
# * g       : corresponding evaluations of the limit-state function g(x)
# ---------------------------------------------------------------------------
# References:
# 1."Bayesian post-processor and other enhancements of Subset Simulation
#    for estimating failure probabilites in high dimension"
#    Zuev, Beck, Au, Katafygiotis (2012)
# ---------------------------------------------------------------------------
"""

import time as timer
import numpy as np

import random

import utilities.seed_selector as ss 

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def mp(N, LSF, sampler, sample_marg_PDF_list):
    m_max = 1e7

    # get dimension
    d = len(sample_marg_PDF_list)

    # initialization
    theta = np.zeros((N, d), float)
    g     = np.zeros(N, float)
    acc   = 0

    seed_id_list = [i for i in range(0, N)]
    g_list       = []

    # MCS sampling
    for i in range(0, N):
        for k in range(0, d):
            theta[i, k] = sample_marg_PDF_list[k]()

        g[i] = LSF(theta[i, :])
        g_list.append(g[i])

    m = 0

    while np.max(g) > 0 and m < m_max:
        # get index of smallest g
        id_min = np.argmax(g)
        #print('id_min =', id_min)

        # get theta_seed randomly from all theta (despite of theta[id_min])
        seed_id_list_tmp = seed_id_list.copy()
        seed_id_list_tmp.pop(id_min)
        seed_id          = random.choice(seed_id_list_tmp)
        theta_seed       = theta[seed_id]

        # sampling
        theta_temp, g_temp = sampler.sample_markov_chain(theta_seed, 1, LSF, g[id_min])

        # count acceptance rate
        if g[id_min] != g_temp:
            acc = acc + 1

        theta[id_min] = theta_temp
        g[id_min]     = g_temp
        g_list.append(g_temp)

        m = m + 1
        print('m:', m, '| g =', g_temp)

    pf_hat = (1 - 1/N)**m
    acc_rate = acc / m

    return pf_hat, theta, g_list, acc_rate, m


# ---------------------------------------------------------------------------
# WITH SEED SELECTION
# ---------------------------------------------------------------------------
def mp_with_seed_selection(N, LSF, sampler, sample_marg_PDF_list, seed_selection_strategy):
    m_max = 1e7

    # get dimension
    d = len(sample_marg_PDF_list)

    # initialization
    theta = np.zeros((N, d), float)
    g     = np.zeros(N, float)
    acc   = 0

    g_list            = []
    theta_list   = []
    for i in range(0, N):
        new_list = []
        theta_list.append(new_list)
    
    # initialize seed selector
    seed_selector = ss.SeedSelector(seed_selection_strategy, N)

    # MCS sampling
    for i in range(0, N):
        theta_temp = np.zeros(d)
        for k in range(0, d):
            theta_temp[k] = sample_marg_PDF_list[k]()
            
        theta[i, :] = theta_temp
        theta_list[i].append(theta_temp)
        g[i] = LSF(theta_temp)
        g_list.append(g[i])


    m_array = np.zeros(N, int)
    m = 0

    while np.max(g) > 0 and m < m_max:
        # get index of smallest g
        id_min = np.argmax(g)

        # get theta_seed randomly from all theta (despite of theta[id_min])
        seed_id = seed_selector.get_seed_id(id_min)
        theta_seed = theta[seed_id]

        # sampling
        theta_temp, g_temp = sampler.sample_markov_chain(theta_seed, 1, LSF, g[id_min])

        # count acceptance rate
        if g[id_min] != g_temp:
            acc = acc + 1

        theta[id_min] = theta_temp
        g[id_min]     = g_temp
        g_list.append(g_temp)
        theta_list[id_min].append(theta_temp.reshape(d))

        m_array[id_min] = m_array[id_min] + 1
        m = np.sum(m_array)
        print('m:', m, '| g =', g_temp)

    pf_hat = (1 - 1/N)**m
    acc_rate = acc / m

    return pf_hat, theta_list, g_list, acc_rate, m_array


# ---------------------------------------------------------------------------
# ONLY ONE PARTICLE
# ---------------------------------------------------------------------------
def mp_one_particle(N, LSF, sampler, sample_marg_PDF_list):
    m_max = 1e7

    # get dimension
    d = len(sample_marg_PDF_list)

    # initialization
    theta = np.zeros((N, d), float)
    g     = np.zeros(N, float)
    acc   = 0

    g_list = []
    m_list = []

    # MCS sampling
    for i in range(0, N):
        for k in range(0, d):
            theta[i, k] = sample_marg_PDF_list[k]()

        g_temp = LSF(theta[i, :])
        g[i] = g_temp
        g_list.append(g_temp)
    

    # move every particle in the failure domain and count the steps (= m)
    for i in range(0, N):
        m = 0
        while g[i] > 0 and m < m_max:
            # sample theta[i] from F(.|g_[i] < g_[i-1]))
            theta_temp, g_temp = sampler.sample_markov_chain(theta[i], 1, LSF, g[i])

            #g_temp = g_temp[0]
            #theta_temp[:] = theta_temp[0,:]

            # count acceptance rate
            if g[i] != g_temp:
                acc = acc + 1

            theta[i] = theta_temp
            g[i]     = g_temp

            # save g-value in list
            g_list.append(g_temp)

            m = m + 1
            print('m:', m, '| g =', g_temp)

        m_list.append(m)

    m_sum = np.sum(np.asarray(m_list))
    pf_hat = (1 - 1/N)**m_sum
    acc_rate = acc/m_sum

    return pf_hat, theta, g_list, acc_rate, m_list
