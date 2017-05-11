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

# ---------------------------------------------------------------------------
# Subset Simulation function
def subsetsim(p0, n_samples_per_level, d, sample_marg_PDF, f_marg_PDF, sample_prop_PDF, f_prop_PDF, LSF):
    # initialization and constants
    max_it  = 20
    theta   = []
    g       = []
    #g       = np.zeros((N), float)

    Nf      = np.zeros(max_it)
    b       = np.zeros(max_it)

    Nc      = int(n_samples_per_level*p0) # number of chains / number of seeds per level 
    Ns      = int(1/p0) # number of samples per chain / number of samples per 

    
    print('\n> > Start STEP 0 : Monte Carlo Simulation')
    startTime = timer.time()

    # sample initial step (MCS)
    j       = 0 # set j = 0 (number of conditional level)
    theta0  = sample_marg_PDF((n_samples_per_level, d))
    g0      = np.zeros((n_samples_per_level), float)

    for i in range(0, n_samples_per_level):
        g0[i] = LSF(theta0[i, :])  # evaluate theta0
        if (g0[i] <= 0):
            Nf[j] += 1
    print('> > Nf =', Nf[j], '/', n_samples_per_level)
    print('> > End STEP 0 : Time needed =', round(timer.time() - startTime, 2), 's')
    theta.append(theta0)
    g.append(g0)

    last_loop = False

    # Subset Simulation steps
    #while last_loop != True:
        #if Nf[j] >= Nc:
        #    last_loop = True
    while Nf[j] < Nc:
        j += 1 # move to next conditional level

        print('\n> > Start STEP', j, ': Subset Simulation')
        startTime = timer.time()

        sortTime = timer.time()
        # sort {g(i)} : g(i1) <= g(i2) <= ... <= g(iN)
        g_prime = np.sort(g0) # sorted g
        idx = sorted(range(len(g0)), key=lambda x: g0[x])

        # order samples according to the previous order
        theta_prime = theta0[(idx)] # sorted theta
        print('> > Sorting: Time needed =', round(timer.time() - sortTime, 2), 's')

        thresholdTime = timer.time()
        # compute intermediate threshold level
        # define b(j) = (g(i_(N*p_0) + g(i_(N*p0 + 1)) / 2
        b[j] = 0.5* (g_prime[Nc] + g_prime[Nc + 1])
        print("> > b =", b[j])
        print('> > Computing Threshold: Time needed =', round(timer.time() - thresholdTime, 2), 's')

        seedTime = timer.time()
        # select seeds for the MCMC sampler
        theta_seed = theta_prime[:Nc, :]
        theta_seed = np.random.permutation(theta_seed) # shuffle to prevent bias
        print('> > Selecting seeds: Time needed =', round(timer.time() - seedTime, 2), 's')

        # re-initialize theta0 and g0 to prevent old values
        theta0  = np.zeros((n_samples_per_level, d), float)
        g0      = np.zeros(n_samples_per_level, float)

        sampleTime = timer.time()
        for k in range(0, Nc):
            # generate states of Markov chain using MMA/MMH
            theta_temp, g_temp = mmh.modified_metropolis(theta_seed[k, :], Ns, f_marg_PDF, sample_prop_PDF, f_prop_PDF, LSF, b[j])
            theta0[Ns*(k):Ns*(k+1), :] = theta_temp[:,:]
            g0[Ns*(k):Ns*(k+1)] = g_temp[:]
        print('> > Sampling MMH: Time needed =', round(timer.time() - sampleTime, 2), 's')

        theta.append(theta0)
        g.append(g0)

        countTime = timer.time()
        # count failure samples
        for i in range(0, n_samples_per_level):
            if g0[i] <= 0:
                Nf[j] += 1
        print('> > Counting failure samples: Time needed =', round(timer.time() - countTime, 2), 's')

        print('> > Nf =', Nf[j], '/', n_samples_per_level)
        print('> > End STEP', j, ': Time needed =', round(timer.time() - startTime, 2), 's')

    # estimate of p_F
    p_F_SS = (p0**(j-1)) * Nf[j-1]/n_samples_per_level

    return p_F_SS, theta, g
