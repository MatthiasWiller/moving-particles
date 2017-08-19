"""
# ---------------------------------------------------------------------------
# Adaptive Conditional Sampling algorithm function
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

import time as timer
import numpy as np

class AdaptiveCondSampling:
    def __init__(self, sample_marg_PDF, pa):
        self.sample_marg_PDF = sample_marg_PDF
        self.pa              = pa
        self.lambda_0        = 0.6


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
        # optimal acceptance rate
        a_star       = 0.44

        # number of chains for adaption
        Na          = int(self.pa*Ns)
        lambda_t    = np.zeros(int(Nc/Na), float)
        lambda_t[0] = self.lambda_0

        # get dimension
        d = np.size(theta_seed, axis=1)

        # get sigma of theta_seed
        sigma_tilde = np.std(theta_seed, axis=0)

        # initialization
        sigma_k = np.zeros((d), float)
        rho_k   = np.zeros((d), float)

        # set sigma_k = min(1, lambda0*sigma)
        sigma_k  = np.minimum(np.ones(d), lambda_t[0]*sigma_tilde)
        # set rho_k = sqrt(1.0 - sigma_k**2)
        rho_k    = np.sqrt(np.ones(d) - sigma_k**2)

        # shuffle seeds to prevent bias
        theta_seed = np.random.permutation(theta_seed)

        # initialization
        theta_list = []
        g_list     = []
        a_list     = []

        # empty list to store the acc/rej-values up until adaptation
        a_bar      = []

        for i in range(0, Nc):
            msg = "> > Sampling Level ... [" + repr(int(i/Nc*100)) + "%]"
            print(msg)

            # generate states of Markov chain
            theta_temp, g_temp, a_temp = self.sample_markov_chain(theta_seed[i, :], Ns, LSF, b, sigma_k, rho_k)

            # save Markov chain in list
            theta_list.append(theta_temp)
            g_list.append(g_temp)
            a_list.append(a_temp)
            a_bar.append(a_temp)

            if i != 0 and np.mod(i, Na) == 0:
                # compute number of adaptation step
                t       = int(np.floor(i/Na))

                # get mean accaptence rate since last adaptation step
                a_array = np.asarray(a_bar).reshape(-1)
                a_hat_t = np.mean(a_array)

                # reset a_bar
                a_bar   = []

                # compute lambda_t for this adaptation step
                lambda_t[t] = np.exp(np.log(lambda_t[t-1]) + (a_hat_t - a_star)/np.sqrt(t))
                # compute sigma_k and rho_k
                sigma_k     = np.minimum(np.ones(d), lambda_t[t]*sigma_tilde)
                rho_k       = np.sqrt(np.ones(d) - sigma_k**2)

        # save last lambda_t for next level
        self.lambda_0 = lambda_t[t]

        # convert theta_list and g_list to np.array()
        theta_array = np.asarray(theta_list).reshape((-1, d))
        g_array     = np.asarray(g_list).reshape(-1)

        # output
        return theta_array, g_array

    def sample_markov_chain(self, theta0, Ns, LSF, b, sigma_k, rho_k):
        # get dimension
        d           = np.size(theta0)

        # initialize theta and g(x)
        theta       = np.zeros((Ns, d), float)
        theta[0, :] = theta0
        g           = np.zeros((Ns), float)
        g[0]        = LSF(theta0)

        a           = np.zeros(Ns, float)
        a[0]        = 1 # first sample is always accepted (= seed)

        for i in range(1, Ns):
            theta_star = np.zeros(d, float)
            # generate a candidate state xi:
            for k in range(0, d):
                # compute sigma and mu from rho_k and theta[i-1]
                sigma_cond = np.sqrt(1 - rho_k[k]**2)
                mu_cond    = rho_k[k] * theta[i-1, k]

                # sample the candidate state
                theta_star[k] = np.random.normal(mu_cond, sigma_cond, 1)

            # check whether theta_star is in Failure domain (system analysis) and accept/reject it
            g_star = LSF(theta_star)
            if g_star <= b:
                # in failure domain -> accept
                theta[i, :] = theta_star
                g[i]        = g_star
                a[i]        = 1
            else:
                # not in failure domain -> reject
                theta[i, :] = theta[i-1, :]
                g[i]        = g[i-1]
                a[i]        = 0

        # output
        return theta, g, a
