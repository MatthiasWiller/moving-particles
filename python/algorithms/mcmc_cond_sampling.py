"""
# ---------------------------------------------------------------------------
# Metropolis-Hastings algorithm
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-10
# ---------------------------------------------------------------------------
# References:
# 1."Simulation and the Monte Carlo method"
#    Rubinstein and Kroese (2017)
# ---------------------------------------------------------------------------
"""

import time as timer
import numpy as np

def cond_sampling(initial_theta, n_samples, rho_k, burnInPeriod, lagPeriod):
    print(">==========================================================================")
    print("> Properties of Sampling:")
    print("> Algorithm \t\t= Conditional Sampling")
    print("> Number of samples \t=", n_samples)
    print("> Lag-Period \t\t=", lagPeriod)
    print("> Burning-In-Period \t=", burnInPeriod)

    print("\n\n> Starting sampling")
    startTime = timer.time()

    d = np.size(initial_theta)
    N = int((n_samples + burnInPeriod)*lagPeriod)

    # initialize theta
    theta = np.zeros((d, N), float)
    theta[:, 0] = initial_theta[:, 0]

    sigma_cond = np.sqrt(1 - rho_k**2)

    # initialization
    i = 1

    # loop
    while i < N:
        msg = 'Sample ' + repr(i) + '/' + repr(N) + '...\n'
        print(msg, sep=' ', end='', flush=True)

        theta_star = np.zeros(d, float)
        # sample theta_star from proposal_PDF
        for k in range(0, d):
            # sample the candidate state
            mu_cond       = rho_k * theta[k, i-1]
            theta_star[k] = np.random.normal(mu_cond, sigma_cond, 1)
        
        theta[:, i] = theta_star
        i+=1

    # reduce samples with lagPerdiod (thinning)
    if lagPeriod != 1:
        theta_red = np.zeros((d, n_samples + burnInPeriod), float)

        for i in range(0, n_samples + burnInPeriod):
            theta_red[:, i] = theta[:, i*lagPeriod]

        theta = theta_red

    # apply burn-in-period
    theta = theta[:, burnInPeriod:]
    
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
    #     
    print("> Time needed for testing =", round(timer.time() - startTime,2), "s")

    print("\n> Test results:")
    print("> rel_eps_mu \t\t=", round(rel_eps_mu,5))

    print(">==========================================================================")
    return theta
