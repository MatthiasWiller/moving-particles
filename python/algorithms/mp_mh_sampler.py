"""
# ---------------------------------------------------------------------------
# Sampling with a Metropolis Hastings Transition Kernel
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

class MHSampler:
    def __init__(self, burnin, sigma, f_marg_PDF):
        self.burnin          = burnin
        self.sigma_sampling  = sigma
        self.f_marg_PDF      = f_marg_PDF

    def get_next_sample(self, theta0, g0, LSF):
        # get dimension
        d       = len(theta0)

        # initialization
        theta_temp = theta0
        g_temp     = g0

        b = self.burnin
        while b > 0:
            # initialization
            # w = np.zeros(d, float)
            # for k in range(0, d):
            #     w = np.random.randn(1)
            w = np.random.randn(d)

            theta_star = theta_temp + self.sigma_sampling * w

            # generate a candidate state xi:
            for k in range(0, d):
                # compute acceptance ratio

                alpha = (self.f_marg_PDF[k](theta_star[k]))/ \
                        (self.f_marg_PDF[k](theta_temp[k]))

                r     = np.minimum(alpha, 1)

                # accept or reject xi by setting ...
                if np.random.uniform(0, 1) <= r:
                    # accept
                    theta_star[k] = theta_star[k]
                else:
                    # reject
                    theta_star[k] = theta_temp[k]

            # check whether theta_star is better than theta0
            g_star = LSF(theta_star)
            if g_star < g0:
                # in failure domain -> accept
                theta_temp = theta_star
                g_temp     = g_star

            b = b - 1

        return theta_temp, g_temp
