"""
# ---------------------------------------------------------------------------
# Modified Metropolis Hastings algorithm function
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

class ModifiedMetropolisHastings:
    def __init__(self, sample_marg_PDF, f_marg_PDF, sample_prop_PDF, f_prop_PDF,):
        self.f_marg_PDF      = f_marg_PDF
        self.sample_marg_PDF = sample_marg_PDF
        self.f_prop_PDF      = f_prop_PDF
        self.sample_prop_PDF = sample_prop_PDF

    def get_mcs_samples(self):
        return self.sample_marg_PDF

    def sample_markov_chain(self, theta0, N, LSF, b):
        #startTime = timer.time()

        # get dimension
        d = np.size(theta0)

        # initialize theta and g(x)
        theta       = np.zeros((N, d), float)
        theta[0, :] = theta0
        g           = np.zeros((N), float)
        g[0]        = LSF(theta0)

        xi          = np.zeros((d), float)

        for i in range(1, N):
            # generate a candidate state xi:
            for k in range(0, d):
                # sample xi from proposal_PDF
                xi[k] = self.sample_prop_PDF(theta[i-1, k])

                # compute acceptance ratio

                # alpha = (p(y) * q(y,x)) /   =   (p(y) * g(y)) /
                #         (p(x) * q(x,y))         (p(x) * g(x))
                alpha = (self.f_marg_PDF(xi[k])          * self.f_prop_PDF(theta[i-1, k], xi[k]))/ \
                        (self.f_marg_PDF(theta[i-1, k])  * self.f_prop_PDF(xi[k], theta[i-1, k]))

                r     = np.minimum(alpha, 1)

                # accept or reject xi by setting ...
                if np.random.uniform(0, 1) <= r:
                    # accept
                    xi[k] = xi[k]
                else:
                    # reject
                    xi[k] = theta[i, k]

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

        # output
        #print("> > > Time needed for MMH =", round(timer.time() - startTime, 2), "s")
        return theta, g
