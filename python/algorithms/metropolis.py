"""
# ---------------------------------------------------------------------------
# Metropolis algorithm function
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
# * initial_theta   : initial sample to run the Markov chain
# * n_samples       : total number of simulated samples
# * target_PDF      : function to be sampled
# * proposal_PDF    : proposal PDF function
# * burnInFraction  : fraction of burn-in samples
# * lagPeriod       : to perform thinning of the Markov chain sequence
# ---------------------------------------------------------------------------
# Output:
# * theta : samples distributed according to 'target_PDF'
# ---------------------------------------------------------------------------
# References:
# 1."Markov chain Monte Carlo (MCMC)"
#    Kevnin P. Murphy (2006)
#
# 2."Markov Chain Monte Carlo and Gibbs Sampling"
#    B. Walsh (2004)
# ---------------------------------------------------------------------------
"""

import time as timer

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Metropolis algorithm
def metropolis(initial_theta, n_samples, target_PDF, proposal_PDF, burnInFraction, lagPeriod):
    print(">==========================================================================")
    print("> Properties of Sampling:")
    print("> Algorithm \t\t= Metropolis")
    print("> Number of samples \t=", n_samples)
    print("> Lag-Period \t\t=", lagPeriod)
    print("> Burning-In-Fraction \t=", burnInFraction)
    
    print("\n> Starting sampling")
    startTime = timer.time()

    # initialize N
    burnInPeriod    = int (n_samples * burnInFraction)
    N               = (n_samples + burnInPeriod)*lagPeriod

    # initialize theta
    theta       = np.zeros(N, float)
    theta[0]    = initial_theta

    # initialization
    i                  = 1
    n_accepted_samples = 0

    # loop
    while i < N:
        # sample theta_star from proposal_PDF
        theta_star = proposal_PDF()

        # alpha = p(y) / p(x)
        alpha = target_PDF(theta_star) / target_PDF(theta[i-1])

        r = np.minimum(alpha, 1)

        # accept or reject sample
        if (np.random.uniform(0, 1) <= r):
            theta[i] = theta_star
            n_accepted_samples +=1
        else:
            theta[i] = theta[i-1]
            
        i+=1
    
    
    # reduce samples with lagPeriod
    if lagPeriod != 1:
        theta_red = np.zeros((n_samples + burnInPeriod), float)

        for i in range(0, n_samples + burnInPeriod):
            theta_red[i] = theta[i*lagPeriod]

        theta = theta_red

    # apply burn-in-period
    theta = theta[burnInPeriod:]

    print("> Time needed for sampling =",round(timer.time() - startTime,2),"s")

    startTime = timer.time()

    print("\n> Starting tests")

    # TESTS

    # geweke-test
    start_fractal   = int ((n_samples-burnInPeriod) * 0.1)
    end_fractal     = int ((n_samples-burnInPeriod) * 0.5)
    
    mu_start        = np.mean(theta[:start_fractal])
    mu_end          = np.mean(theta[end_fractal:])

    rel_eps_mu      = (mu_start - mu_end)/ mu_end

    # acceptance rate
    acceptance_rate = n_accepted_samples/N


    print("> Time needed for testing =",round(timer.time() - startTime,2),"s")

    print("\n> Test results:")
    print("> rel_eps_mu \t\t=", round(rel_eps_mu,5))
    print("> acceptance rate \t=", round(acceptance_rate,4), "\t(optimal if in [0.20; 0.44])")

    print(">==========================================================================")
    return theta
