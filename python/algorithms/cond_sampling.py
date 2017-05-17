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
# * sample_marg_PDF : function to sample from marginal pdf
# * LSF             : limit state function
# * b               : threshold level of the limit state function
# * rho_k           : correlation coefficient between two sample-states
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

class CondSampling:
    def __init__(self, sample_marg_PDF, sample_cond_PDF, rho_k):
        self.sample_marg_PDF = sample_marg_PDF
        self.sample_cond_PDF = sample_cond_PDF
        self.rho_k           = rho_k

    def get_mcs_samples(self):
        return self.sample_marg_PDF

    def sample_markov_chain(self, theta0, N, LSF, b):
        #startTime = timer.time()

        # get dimension
        d           = np.size(theta0)

        # initialize theta and g(x)
        theta       = np.zeros((N, d), float)
        theta[0, :] = theta0
        g           = np.zeros((N), float)
        g[0]        = LSF(theta0)

        # compute sigma from correlation parameter rho_k
        sigma       = np.sqrt(1 - self.rho_k**2)

        for i in range(1, N):
            theta_star = np.zeros(d, float)
            # generate a candidate state xi:
            for k in range(0, d):
                # sample the candidate state
                mu = self.rho_k * theta[i-1, k]
                theta_star[k] = self.sample_cond_PDF(mu, sigma)

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
