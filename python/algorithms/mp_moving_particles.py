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
# Version 2017-05
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

def mp(N, LSF, sampler, sample_marg_PDF_list):
    m_max = 1e7

    # get dimension
    d = len(sample_marg_PDF_list)

    # initialization
    theta = np.zeros((N, d), float)
    g     = np.zeros(N, float)
    acc   = 0

    # MCS sampling
    for i in range(0, N):
        for k in range(0, d):
            theta[i, k] = sample_marg_PDF_list[k]()

        g[i] = LSF(theta[i, :])

    m = 0

    while np.max(g) > 0 and m < m_max:
        # get index of smallest g
        id_min = np.argmax(g)

        # sampling
        theta_temp, g_temp = sampler.get_next_sample(theta[id_min], g[id_min], LSF)

        # count acceptance rate
        if g[id_min] != g_temp:
            acc = acc + 1

        theta[id_min] = theta_temp
        g[id_min]     = g_temp

        m = m + 1
        print('m:', m, '| g =', g_temp)

    pf_hat = (1 - 1/N)**m

    return pf_hat, theta, g, acc





def mp_one_particle(N, LSF, sampler, sample_marg_PDF_list):
    m_max = 1e7

    # get dimension
    d = len(sample_marg_PDF_list)

    # initialization
    theta = np.zeros((N, d), float)
    g     = np.zeros(N, float)
    acc   = 0

    # MCS sampling
    for i in range(0, N):
        for k in range(0, d):
            theta[i, k] = sample_marg_PDF_list[k]()

        g[i] = LSF(theta[i, :])

    m_list = []

    # move every particle in the failure domain and count the steps (= m)
    for i in range(0, N):
        m = 0
        while g[i] > 0 and m < m_max:
            # sample theta[i] from F(.|g_[i] > g_[i-1]))
            theta_temp, g_temp = sampler.get_next_sample(theta[i], g[i], LSF)

            # count acceptance rate
            if g[i] != g_temp:
                acc = acc + 1

            theta[i] = theta_temp
            g[i]     = g_temp

            m = m + 1
            print('m:', m, '| g =', g_temp)

        m_list.append(m)

    m_sum = np.sum(np.asarray(m_list))
    pf_hat = (1 - 1/N)**m_sum
    acc_rate = acc/m_sum

    return pf_hat, theta, g, acc_rate, m_list
