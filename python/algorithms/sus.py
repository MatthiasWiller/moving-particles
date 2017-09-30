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
def subsetsim(p0, n_samples_per_level, LSF, sampler):
    # initialization and constants
    max_it  = 20
    theta   = []
    g       = []

    Nf      = np.zeros(max_it)
    b       = np.zeros(max_it)

    Nc      = int(n_samples_per_level*p0) # number of chains / number of seeds per level 
    Ns      = int(1/p0) # number of samples per chain / number of samples per 


    # print('\n> > Start LEVEL 0 : Monte Carlo Simulation')
    startTime = timer.time()

    # sample initial step (MCS)
    j       = 0 # set j = 0 (number of conditional level)

    # sample initial step (MCS)
    theta0, g0 = sampler.sample_mcs_level(n_samples_per_level, LSF)

    for i in range(0, n_samples_per_level):
        #g0[i] = LSF(theta0[i, :])  # evaluate theta0
        if g0[i] <= 0:
            Nf[j] += 1

    # print('> > Nf =', Nf[j], '/', n_samples_per_level)
    # print('> > End LEVEL 0 : Time needed =', round(timer.time() - startTime, 2), 's')
    theta.append(theta0)
    g.append(g0)

    # loop while pF <= Nc/N
    while Nf[j] < Nc:
        j += 1 # move to next conditional level

        # check, if Simulation has reached max. level
        if j >= max_it:
            print('\n> > ERROR: Reached max. Levels without converging to the failure domain!')

        # print('\n> > Start LEVEL', j, ': Subset Simulation')
        startTime = timer.time()

        # sort {g(i)} : g(i1) <= g(i2) <= ... <= g(iN)
        g_prime = np.sort(g0) # sorted g
        idx = sorted(range(len(g0)), key=lambda x: g0[x])

        # order samples according to the previous order
        theta_prime = theta0[(idx)] # sorted theta

        # compute intermediate threshold level
        # define b(j) = (g(i_(N*p_0) + g(i_(N*p0 + 1)) / 2
        b[j] = np.percentile(g_prime, p0*100)
        # print("> > b =", b[j])

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

        # print('> > Nf =', Nf[j], '/', n_samples_per_level)
        # print('> > End LEVEL', j, ': Time needed =', round(timer.time() - startTime, 2), 's')

    # estimate of p_F
    p_F_SS = (p0**(j)) * Nf[j]/n_samples_per_level

    return p_F_SS, theta, g
