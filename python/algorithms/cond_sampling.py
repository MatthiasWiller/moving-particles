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
# * Ns              : number of samples of Markov-chain (including seed)
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

import numpy as np

class CondSampling:
    def __init__(self, sample_marg_PDF, rho_k, burnin=0):
        self.sample_marg_PDF = sample_marg_PDF
        self.rho_k           = rho_k
        self.T               = burnin

    def sample_mcs_level(self, n_samples_per_level, LSF):
        # get dimension
        d       = len(self.sample_marg_PDF)

        # initialize theta0 and g0
        theta0  = np.zeros((n_samples_per_level, d), float)
        g0      = np.zeros(n_samples_per_level, float)


        for i in range(0, n_samples_per_level):
            # sample theta0
            for k in range(0, d):
                theta0[i, k] = self.sample_marg_PDF[k]()

            # evaluate theta0
            g0[i] = LSF(theta0[i, :])

        # output
        return theta0, g0


    def sample_subsim_level(self, theta_seed, Ns, Nc, LSF, b):
        # get dimension
        d       = np.size(theta_seed, axis=1)

        # initialization
        theta_list = []
        g_list     = []

        # shuffle seeds to prevent bias
        theta_seed = np.random.permutation(theta_seed)

        for k in range(0, Nc):
            msg = "> > Sampling Level ... [" + repr(int(k/Nc*100)) + "%]"
            print(msg)

            # generate states of Markov chain
            theta_temp, g_temp = self.sample_markov_chain(theta_seed[k, :], Ns, LSF, b)

            # save Markov chain in list
            theta_list.append(theta_temp)
            g_list.append(g_temp)

         # convert theta_list and g_list to np.array()
        theta_array = np.asarray(theta_list).reshape((-1, d))
        g_array     = np.asarray(g_list).reshape(-1)

        # output
        return theta_array, g_array

    def sample_markov_chain(self, theta0, Ns, LSF, b):
        # get dimension
        d           = np.size(theta0)

        # initialize theta and g(x)
        theta       = np.zeros((self.T+Ns, d), float)
        theta[0, :] = theta0
        g           = np.zeros((self.T+Ns), float)
        g[0]        = LSF(theta0)

        # compute sigma from correlation parameter rho_k
        sigma_cond  = np.sqrt(1 - self.rho_k**2)


        for i in range(1, self.T+Ns):
            theta_star = np.zeros(d, float)
            # generate a candidate state xi:
            for k in range(0, d):
                # sample the candidate state
                mu_cond       = self.rho_k * theta[i-1, k]
                theta_star[k] = np.random.normal(mu_cond, sigma_cond, 1)

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
        
        # apply burn-in
        theta = theta[self.T:, :]
        g     = g[self.T:]

        # output
        return theta, g
