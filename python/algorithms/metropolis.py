"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def metropolis(initial_theta, n_samples, target_PDF, proposal_PDF, burningInFraction, logPeriod):
    print("START Metropolis-sampling")

    # set seed
    np.random.seed(0)

    # initialize theta
    theta = np.zeros((n_samples*logPeriod), float)
    theta[0] = initial_theta

    # initialization
    i = 1
    n_accepted_samples = 0

    # loop
    while i < n_samples*logPeriod:
        # sample theta_star from proposal_PDF
        theta_star = proposal_PDF()
        alpha = np.minimum(target_PDF(theta_star)/ target_PDF(theta[i-1]), 1)
        
        # accept or reject sample
        if (np.random.random([1]) <= alpha):
            theta[i] = theta_star
            # print("accept!\n")
            n_accepted_samples +=1
        else:
            theta[i] = theta[i-1]
            # print("reject!\n")
            
        i+=1
    
    
    # reduce samples with logPerdiod
    theta_red = np.zeros((n_samples), float)

    for i in range(0,n_samples):
        theta_red[i] = theta[i*logPeriod]
    
    theta = theta_red

    # apply burning-in-period
    burningInPeriod = int (n_samples * burningInFraction)
    theta = theta[burningInPeriod:]
    
    # TESTS

    # genervece-test
    start_fractal = int ((n_samples-burningInPeriod) * 0.1)
    end_fractal = int ((n_samples-burningInPeriod) * 0.5)
    
    mu_start = np.mean(theta[:start_fractal])
    mu_end = np.mean(theta[end_fractal:])

    rel_eps_mu = (mu_start - mu_end)/ mu_end
    print("rel_eps_mu = ", rel_eps_mu)

    # acceptance rate
    print("acceptance rate = ", n_accepted_samples/(n_samples*logPeriod), " (optimal if between [0.20;0.44])")

    print("END Metropolis-method")
    return theta
