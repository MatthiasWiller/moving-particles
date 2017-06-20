"""
# ---------------------------------------------------------------------------
# Sampling with a Transition Kernel 
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

class CondSampler:
    def __init__(self, burnin, rho_k):
        self.burnin = burnin
        self.rho_k  = rho_k

    def get_next_sample(self, theta0, g0, LSF):
        # get dimension
        d       = len(theta0)

        # initialization
        theta_temp = theta0
        g_temp     = g0
        
        sigma_cond = np.sqrt(1 - self.rho_k**2)

        b = self.burnin
        while b > 0:
            mu_cond = self.rho_k * theta_temp

            # initialization
            theta_star = np.zeros(d, float)

            # sample from gaussian with cond. parameters
            for k in range(0, d):
                theta_star[k] = np.random.normal(mu_cond[k], sigma_cond, 1)

            g_star     = LSF(theta_star)

            # check if new sample is better than old sample
            if g0 > g_star:
                # accept
                #print('sample got accepted')
                #print('g_old =', g0, '--> g_new =', g_star)
                theta_temp = theta_star
                g_temp     = g_star

            #else:
                # reject
                #print('-> sample got rejected')

            b = b - 1

        return theta_temp, g_temp
