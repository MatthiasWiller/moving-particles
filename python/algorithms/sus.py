"""
Author: Matthias Willer 2017
"""

import numpy as np

import time as timer

import algorithms.modified_metropolis as mmh

# p0: conditional failure probability
# n_samples_per_level: number of samples per conditional level
# target_PDF
# proposal_PDF
# LSF: limit state function g(x)

def subsetsim(p0, n_samples_per_level, d, marginal_PDF, sample_prop_PDF, f_prop_PDF, LSF):
    # initialization and constants
    max_it  = 20
    theta   = []
    g       = []
    #g       = np.zeros((N), float)

    Nf      = np.zeros(max_it)
    b       = np.zeros(max_it)

    n_seeds_per_level   = int(n_samples_per_level*p0)
    n_samples_per_seed  = int(1/p0) # including the seed

    
    print('\n> > Start STEP 0 : Monte Carlo Simulation')
    startTime = timer.time()

    # sample initial step (MCS)
    j = 0 # set j = 0 (number of conditional level)
    theta0  = np.random.randn(n_samples_per_level, d)
    g0      = np.zeros((n_samples_per_level), float)

    for i in range(0, n_samples_per_level):
        g0[i] = LSF(theta0[i, :])  # evaluate theta0
        if (g0[i] <= 0):
            Nf[j] += 1
    print('> > Nf =', Nf[j], '/', n_samples_per_level)
    print('> > End STEP 0 : Time needed =', round(timer.time() - startTime, 2), 's')
    theta.append(theta0)
    g.append(g0)

    # Subset Simulation steps
    while Nf[j] < n_seeds_per_level:
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
        # define b(j) = (g(i_(N - N*p_0) + g(i_(N - N*p0 + 1)) / 2
        #b[j] = (g_prime[n_samples_per_level- n_seeds_per_level] + g_prime[n_samples_per_level - n_seeds_per_level + 1]) /2
        b[j] = (g_prime[n_seeds_per_level] + g_prime[n_seeds_per_level + 1]) /2
        print("> > b =", b[j])
        print('> > Computing Threshold: Time needed =', round(timer.time() - thresholdTime, 2), 's')

        seedTime = timer.time()
        # select seeds for the MCMC sampler
        #theta_seed = theta_prime[-n_seeds_per_level:, :]
        theta_seed = theta_prime[:n_seeds_per_level, :]
        theta_seed = np.random.permutation(theta_seed) # shuffle to prevent bias
        print('> > Selecting seeds: Time needed =', round(timer.time() - seedTime, 2), 's')

        theta_star = []
        sampleTime = timer.time()
        for k in range(0, n_seeds_per_level):
            # generate states of Markov chain using MMA/MMH
            theta_temp = mmh.modified_metropolis(theta_seed[k, :], n_samples_per_seed, marginal_PDF, sample_prop_PDF, f_prop_PDF, LSF, b[j])
            theta_star.append(theta_temp)
        print('> > Sampling MMH: Time needed =', round(timer.time() - sampleTime, 2), 's')

        renumberTime = timer.time()
        theta0 = np.zeros((n_samples_per_level, d), float)

        # renumber theta(j,i,k) ...
        for k in range(0, n_seeds_per_level):
           theta0[n_samples_per_seed*(k):n_samples_per_seed*(k+1), :] = theta_star[k][:, :]
        theta.append(theta0)
        print('> > Renumber Samples: Time needed =', round(timer.time() - renumberTime, 2), 's')

        countTime = timer.time()
        # count failure samples
        for i in range(0, n_samples_per_level):
            g0[i] = LSF(theta0[i, :])
            if g0[i] <= 0:
                Nf[j] += 1
        print('> > Counting failure samples: Time needed =', round(timer.time() - countTime, 2), 's')

        print('> > Nf =', Nf[j], '/', n_samples_per_level)
        print('> > End STEP', j, ': Time needed =', round(timer.time() - startTime, 2), 's')

    # estimate of p_F
    p_F_SS = (p0**j) * Nf[j]/n_samples_per_level

    return p_F_SS, theta
