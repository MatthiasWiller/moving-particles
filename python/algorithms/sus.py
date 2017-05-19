"""
# ---------------------------------------------------------------------------
# Subset Simulation algorithm function
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
# * p0                  : conditional failure probability
# * n_samples_per_level : number of samples per conditional level
# * d                   : number of dimensions
# * sample_marg_PDF     : function to sample from marginal pdf
# * f_marg_PDF          : marginal pdf
# * sample_prop_PDF     : function to sample from proposal pdf
# * f_prop_PDF          : proposal pdf
# * LSF                 : limit state function
# * sampler             : sampling algorithm (cs = Cond. Sampling,
#                         mmh = Modified Metropolis Hastings)
# ---------------------------------------------------------------------------
# Output:
# * p_F_SS  : estimator of the failure probability
# * theta   : list of samples distributed according to 'marginal pdf'
# * g       : list of corresponding evaluations of the limit-state function g(theta)
# ---------------------------------------------------------------------------
# References:
# 1."Bayesian post-processor and other enhancements of Subset Simulation
#    for estimating failure probabilites in high dimension"
#    Zuev, Beck, Au, Katafygiotis (2012)
# ---------------------------------------------------------------------------
"""

import time as timer
import numpy as np

import algorithms.modified_metropolis as mmh
import algorithms.cond_sampling as cs

# ---------------------------------------------------------------------------
# Subset Simulation function
def subsetsim(p0, n_samples_per_level, d, LSF, sampler):
    # initialization and constants
    max_it  = 20
    theta   = []
    g       = []

    Nf      = np.zeros(max_it)
    b       = np.zeros(max_it)

    Nc      = int(n_samples_per_level*p0) # number of chains / number of seeds per level 
    Ns      = int(1/p0) # number of samples per chain / number of samples per 


    print('\n> > Start LEVEL 0 : Monte Carlo Simulation')
    startTime = timer.time()

    # sample initial step (MCS)
    j       = 0 # set j = 0 (number of conditional level)

    theta0 = sampler.sample_mcs_level((n_samples_per_level, d))

    g0      = np.zeros((n_samples_per_level), float)

    for i in range(0, n_samples_per_level):
        g0[i] = LSF(theta0[i, :])  # evaluate theta0
        if (g0[i] <= 0):
            Nf[j] += 1
    print('> > Nf =', Nf[j], '/', n_samples_per_level)
    print('> > End LEVEL 0 : Time needed =', round(timer.time() - startTime, 2), 's')
    theta.append(theta0)
    g.append(g0)

    # loop while pF <= Nc/N
    while Nf[j] < Nc:
        j += 1 # move to next conditional level

        print('\n> > Start LEVEL', j, ': Subset Simulation')
        startTime = timer.time()

        # sort {g(i)} : g(i1) <= g(i2) <= ... <= g(iN)
        g_prime = np.sort(g0) # sorted g
        idx = sorted(range(len(g0)), key=lambda x: g0[x])

        # order samples according to the previous order
        theta_prime = theta0[(idx)] # sorted theta

        # compute intermediate threshold level
        # define b(j) = (g(i_(N*p_0) + g(i_(N*p0 + 1)) / 2
        b[j] = np.percentile(g_prime, p0*100)
        print("> > b =", b[j])

        # select seeds for the MCMC sampler
        theta_seed = theta_prime[:Nc, :]

        # sample level using the sampler/sampling-method
        theta0, g0 = sampler.sample_subsim_level(theta_seed, Ns, Nc, LSF, b[j])

        theta.append(theta0)
        g.append(g0)

        # count failure samples
        for i in range(0, n_samples_per_level):
            if g0[i] <= 0:
                Nf[j] += 1

        print('> > Nf =', Nf[j], '/', n_samples_per_level)
        print('> > End LEVEL', j, ': Time needed =', round(timer.time() - startTime, 2), 's')

    # estimate of p_F
    p_F_SS = (p0**(j-1)) * Nf[j-1]/n_samples_per_level

    return p_F_SS, theta, g


# ---------------------------------------------------------------------------
# compute cov analytically
def cov_analytical(theta, g, p0, N, pf_sus):
    m   = len(g)         # number of levels of the SubSim
    Nc  = int(p0 * N)    # number of Markov chains (= number of seeds)
    Ns  = int(1/p0)      # number of samples per Markov chain

    # initialization
    p       = np.zeros(m, float)
    b       = np.zeros(m, float)
    delta   = np.zeros(m, float)

    # compute intermediate failure levels
    for j in range(0, m):
        g_sort  = np.sort(g[j])
        b[j]    = np.percentile(g_sort, p0*100)
    #print("> > Last threshold =", b[m-1], "-> is now corrected to 0!")
    b[m-1] = 0    # set last threshold to 0

    # compute coefficient of variation for level 0 (MCS)
    delta[0] = np.sqrt(((1 - p0)/(N * p0)))             # [Ref. 1 Eq (3)]

    # compute coefficient of variation for other levels
    for j in range(1, m):

        # compute indicator function matrix for the failure samples
        I_Fj = np.reshape(g[j] <= b[j], (Ns, Nc))

        # sample conditional probability (~= p0)
        p_j = (1/N)*np.sum(I_Fj[:, :])
        print("> > p_j [", j, "] =", p_j)

        # correlation factor (Ref. 2 Eq. 10)
        gamma = np.zeros(Ns, float)

        # correlation at lag 0
        sums = 0
        for k in range(0, Nc):
            for ip in range(1, Ns):
                sums += (I_Fj[ip, k] * I_Fj[ip, k])   # sums inside [Ref. 1 Eq. (22)]
        R_0 = (1/N)*sums - p_j**2    # autocovariance at lag 0 [Ref. 1 Eq. (22)]
        print("R_0 =", R_0)

        # correlation factor calculation
        R = np.zeros(Ns, float)
        for i in range(0, Ns-1):
            sums = 0
            for k in range(0, Nc):
                for ip in range(0, Ns - (i+1)):
                    sums += (I_Fj[ip, k] * I_Fj[ip + i, k])         # sums inside [Ref. 1 Eq. (22)]
            R[i]     = (1/(N - (i+1)*Nc)) * sums - p_j**2               # autocovariance at lag i [Ref. 1 Eq. (22)]
            gamma[i] = (1 - ((i+1)/Ns)) * (R[i]/R_0)                    # correlation factor [Ref. 1 Eq. (20)]

        gamma_j = 2*np.sum(gamma)                                 # [Ref. 1 Eq. (20)]
        print("gamma_j =", gamma_j)

        delta[j] = np.sqrt(((1 - p_j)/(N * p_j)) * (1 + gamma_j)) # [Ref. 2 Eq. (9)]

    # compute resulting cov
    delta_sus = np.sqrt(np.sum(delta**2))
    return delta_sus
