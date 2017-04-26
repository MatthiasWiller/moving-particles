"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import time as timer


def metropolis_hastings(initial_theta, n_samples, target_PDF, sample_prop_PDF, f_prop_PDF, burnInFraction, lagPeriod):
    print(">==========================================================================")
    print("> Properties of Sampling:")
    print("\n> Algorithm \t\t= Metropolis-Hastings")
    print("\n> Number of samples \t=", n_samples)
    print("\n> Lag-Period \t\t=", lagPeriod)
    print("\n> Burn-In-Fraction \t=", burnInFraction)
    
    print("\n\n> Starting sampling")
    startTime = timer.time()

    # set seed
    np.random.seed(1)

    # initialize theta
    theta = np.zeros((n_samples*lagPeriod), float)
    theta[0] = initial_theta

    # initialization
    i = 1
    n_accepted_samples = 0

    # loop
    while i < n_samples*lagPeriod:
        # sample theta_star from proposal_PDF
        theta_star = sample_prop_PDF(theta[i-1])

        p_old = target_PDF(theta[i-1])
        p_new = target_PDF(theta_star)
        alpha = p_new / p_old

        q_denom = f_prop_PDF(theta[i-1], theta_star) # q(x,y) = g(y)
        q_numer = f_prop_PDF(theta_star, theta[i-1]) # q(y,x) = g(x)
        alpha = alpha * q_numer/q_denom
        
        # alpha(x,y) = min[p(y)/p(x) * q(y,x) / q(x,y), 1]
        r = np.minimum(alpha, 1)

        # accept or reject sample
        if (np.random.random(1) <= r):
            theta[i] = theta_star
            # print("accept!\n")
            n_accepted_samples +=1
        else:
            theta[i] = theta[i-1]
            # print("reject!\n")
            
        i+=1
    
    
    # reduce samples with lagPerdiod (thinning)
    theta_red = np.zeros((n_samples), float)

    for i in range(0, n_samples):
        theta_red[i] = theta[i*lagPeriod]
    
    theta = theta_red

    # apply burn-in-period
    burnInPeriod = int (n_samples * burnInFraction)
    theta = theta[burnInPeriod:]
    
    print("> Time needed for sampling =",round(timer.time() - startTime,2),"s")

    startTime = timer.time()

    print("\n\n> Starting tests")

    # TESTS

    # genervece-test
    start_fractal = int ((n_samples-burnInPeriod) * 0.1)
    end_fractal = int ((n_samples-burnInPeriod) * 0.5)
    
    mu_start = np.mean(theta[:start_fractal])
    mu_end = np.mean(theta[end_fractal:])

    rel_eps_mu = (mu_start - mu_end)/ mu_end
    

    # acceptance rate
    acceptance_rate = n_accepted_samples/(n_samples*lagPeriod)
    
    print("> Time needed for testing =",round(timer.time() - startTime,2),"s")

    print("\n\n> Test results:")
    print("\n> rel_eps_mu \t\t=", round(rel_eps_mu,5))
    print("\n> acceptance rate \t=", round(acceptance_rate,4), "\t(optimal if in [0.20; 0.44])")

    print(">==========================================================================")
    return theta
