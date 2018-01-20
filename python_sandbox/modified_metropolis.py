"""
# ---------------------------------------------------------------------------
# Modified Metropolis Hastings algorithm
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
# 1."Bayesian post-processor and other enhancements of Subset Simulation
#    for estimating failure probabilites in high dimension"
#    Zuev, Beck, Au, Katafygiotis (2012)
# ---------------------------------------------------------------------------
"""

import time as timer
import numpy as np

class ModifiedMetropolisHastings:
    def __init__(self, sample_marg_PDF, f_marg_PDF, proposal_dist, sigma_p=1.0, burnin=0):
        self.f_marg_PDF      = f_marg_PDF
        self.sample_marg_PDF = sample_marg_PDF
        self.T               = burnin

        if proposal_dist == 'uniform':
            self.sample_prop_PDF = lambda mu: np.random.uniform(mu - 1.0, mu + 1.0, 1)

        elif proposal_dist == 'gaussian' or proposal_dist == 'normal':
            self.sample_prop_PDF = lambda mu: np.random.normal(mu, sigma_p, 1)

        else:
            self.sample_prop_PDF = lambda mu: 0


    def sample_mcs_level(self, n_samples_per_level, LSF):
        # get dimension
        d      = len(self.sample_marg_PDF)

        # initialize theta0 and g0
        theta  = np.zeros((n_samples_per_level, d), float)
        g      = np.zeros(n_samples_per_level, float)


        for i in range(0, n_samples_per_level):
            # sample theta0
            for k in range(0, d):
                theta[i, k] = self.sample_marg_PDF[k]()

            # evaluate theta0
            g[i] = LSF(theta[i, :])

        return theta, g


    def sample_subsim_level(self, theta_seed, Ns, Nc, LSF, b):
        # get dimension
        d       = np.size(theta_seed, axis=1)

        # initialize theta0 and g0
        theta0  = np.zeros((Ns*Nc, d), float)
        g0      = np.zeros(Ns*Nc, float)

        for k in range(0, Nc):
            # msg = "> > Sampling Level ... [" + repr(int(k/Nc*100)) + "%]"
            # print(msg)

            # generate states of Markov chain
            theta_temp, g_temp = self.sample_markov_chain(theta_seed[k, :], Ns, LSF, b)

            # save Markov chain in sample array
            theta0[Ns*(k):Ns*(k+1), :]  = theta_temp[:, :]
            g0[Ns*(k):Ns*(k+1)]         = g_temp[:]

        return theta0, g0


    def sample_markov_chain(self, theta0, N, LSF, b):
        # get dimension
        d = np.size(theta0)

        # initialize theta and g(x)
        theta       = np.zeros((self.T + N, d), float)
        theta[0, :] = theta0
        g           = np.zeros((self.T + N), float)
        g[0]        = LSF(theta0)

        xi          = np.zeros((d), float)

        for i in range(1, self.T + N):
            # generate a candidate state xi:
            for k in range(0, d):
                # sample xi from proposal_PDF
                xi[k] = self.sample_prop_PDF(theta[i-1, k])

                # compute accaptence ratio
                alpha = self.f_marg_PDF[k](xi[k])         / \
                        self.f_marg_PDF[k](theta[i-1, k]) 

                r     = np.minimum(alpha, 1)

                # accept or reject xi by setting ...
                if np.random.uniform(0, 1) <= r:
                    # accept
                    xi[k] = xi[k]
                else:
                    # reject
                    xi[k] = theta[i-1, k]

            # check whether xi is in Failure domain (system analysis) and accept or reject xi
            g_temp = LSF(xi)
            if g_temp <= b:
                # in failure domain -> accept
                theta[i, :] = xi
                g[i] = g_temp
            else:
                # not in failure domain -> reject
                theta[i, :] = theta[i-1, :]
                g[i] = g[i-1]
        
        # apply burn-in
        theta = theta[self.T:,:]
        g = g[self.T:]

        return theta, g
