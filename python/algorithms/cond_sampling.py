"""
# ---------------------------------------------------------------------------
# Conditional Sampling algorithm function
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

def cond_sampling(theta0, N, f_marg_PDF, sample_prop_PDF, f_prop_PDF, LSF, b):
    #startTime = timer.time()

    # get dimension
    d           = np.size(theta0)

    # initialize theta and g(x)
    theta       = np.zeros((N, d), float)
    theta[0, :] = theta0
    g           = np.zeros((N), float)
    g[0]        = LSF(theta0)

    # set correlation parameter rho_k
    rho_k       = 0.2
    sigma       = np.sqrt(1 - rho_k**2)

    for i in range(1, N):
        theta_star = np.zeros(d, float)
        # generate a candidate state xi:
        for k in range(0, d):
            # sample theta from proposal_PDF
            #theta_star[k] = sample_prop_PDF(theta[i-1, k])

            # sample the candidate state
            mu = rho_k * theta[i-1, k]
            theta_star[k] = np.random.normal(mu, sigma, 1)

        # check whether theta_star is in Failure domain (system analysis) and accept or reject it
        g_star = LSF(theta_star)
        if g_star <= b:
            # in failure domain -> accept
            theta[i, :] = theta_star
            g[i] = g_star
        else:
            # not in failure domain -> reject
            theta[i, :] = theta[i-1, :]
            g[i] = g[i-1]

    # output
    #print("> > > Time needed for CS =", round(timer.time() - startTime, 2), "s")
    return theta, g
