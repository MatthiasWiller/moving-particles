"""
# ---------------------------------------------------------------------------
# Metropolis-Hastings algorithm function
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
# * target_PDF      : function to be sampled
# * f_prop_PDF      : proposal PDF function
# * sample_prop_PDF : proposal PDF sampling
# * initial_theta   : initial sample to run the Markov chain
# * n_samples       : total number of simulated samples
# * burnInPeriod    : burn-in of samples
# * lagPeriod       : to perform thinning of the Markov chain sequence
# ---------------------------------------------------------------------------
# Output:
# * theta : samples distributed according to 'target_PDF'
# ---------------------------------------------------------------------------
# References:
# 1."Simulation and the Monte Carlo method"
#    Rubinstein and Kroese (2017)
# ---------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import time as timer


def metropolis_hastings(initial_theta, n_samples, target_PDF, sample_prop_PDF, f_prop_PDF, burnInPeriod, lagPeriod):
    print(">==========================================================================")
    print("> Properties of Sampling:")
    print("> Algorithm \t\t= Metropolis-Hastings")
    print("> Number of samples \t=", n_samples)
    print("> Lag-Period \t\t=", lagPeriod)
    print("> Burning-In-Period \t=", burnInPeriod)

    print("\n\n> Starting sampling")
    startTime = timer.time()

    d                   = np.size(initial_theta)
    N                   = (n_samples + burnInPeriod)*lagPeriod

    # initialize theta
    theta               = np.zeros((d, N), float)
    theta[:, 0]         = initial_theta[:, 0]

    # initialization
    i                   = 1
    n_accepted_samples  = 0


    # loop
    while i < N:
        msg = 'Sample ' + repr(i) + '/' + repr(N) + '...\n'
        print(msg, sep=' ', end='', flush=True)

        # sample theta_star from proposal_PDF
        theta_star = sample_prop_PDF(theta[:, i-1])

        # alpha = (p(y) * q(y,x)) /   =   (p(y) * g(y)) /
        #         (p(x) * q(x,y))         (p(x) * g(x))
        alpha = (target_PDF(theta_star)     * f_prop_PDF(theta[:, i-1], theta_star))/ \
                (target_PDF(theta[:, i-1])  * f_prop_PDF(theta_star, theta[:, i-1]))

        # alpha(x,y) = min[p(y)/p(x) * q(y,x) / q(x,y), 1]
        r = np.minimum(alpha, 1)

        # accept or reject sample
        if (np.random.uniform(0, 1, 1) <= r):
            theta[:, i] = theta_star
            n_accepted_samples +=1
        else:
            theta[:, i] = theta[:, i-1]
            
        i+=1
    
    print('Acceptance rate =', n_accepted_samples/N)

    # reduce samples with lagPerdiod (thinning)
    if lagPeriod != 1:
        theta_red = np.zeros((d, n_samples + burnInPeriod), float)

        for i in range(0, n_samples + burnInPeriod):
            theta_red[:, i] = theta[:, i*lagPeriod]

        theta = theta_red

    # apply burn-in-period
    theta           = theta[:, burnInPeriod:]
    
    print("> Time needed for sampling =",round(timer.time() - startTime,2),"s")

    startTime = timer.time()

    print("\n> Starting tests")

    # TESTS

    # Geweke-test
    start_fractal   = int ((n_samples-burnInPeriod) * 0.1)
    end_fractal     = int ((n_samples-burnInPeriod) * 0.5)

    if (d == 1):
        # 1D
        mu_start    = np.mean(theta[0,:start_fractal])
        mu_end      = np.mean(theta[0,end_fractal:])

        rel_eps_mu  = (mu_start - mu_end)/ mu_end
    else:
        # multi-dimensional
        rel_eps_mu_list = np.zeros(d, float)
        for i in range(0,d):
            mu_start    = np.mean(theta[i, :start_fractal])
            mu_end      = np.mean(theta[i, end_fractal:])
            
            rel_eps_mu_list[i] = (mu_start - mu_end)/ mu_end

        rel_eps_mu = np.mean(rel_eps_mu_list)
    

    # acceptance rate
    acceptance_rate = n_accepted_samples/N
    
    print("> Time needed for testing =", round(timer.time() - startTime,2), "s")

    print("\n> Test results:")
    print("> rel_eps_mu \t\t=", round(rel_eps_mu,5))
    print("> acceptance rate \t=", round(acceptance_rate,4), "\t(optimal if in [0.20; 0.44])")

    print(">==========================================================================")
    return theta
