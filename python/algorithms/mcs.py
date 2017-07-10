"""
# ---------------------------------------------------------------------------
# Monte Carlo Simulation function
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-06
# ---------------------------------------------------------------------------
# Input:
# * N               : number of samples
# * sample_marg_PDF : function to sample from marginal pdf
# * LSF             : limit state function
# ---------------------------------------------------------------------------
# Output:
# * pf_mcs      : estimator of the failure probability
# * theta_list  : list of samples distributed according to 'marginal pdf'
# * g_list      : list of corresponding evaluations of the limit-state function g(theta)
# ---------------------------------------------------------------------------
"""

import time as timer
import numpy as np

# ---------------------------------------------------------------------------
# Monte Carlo Simulation function
def mcs(N, sample_marg_PDF_list, LSF):
    # get number of dimensions
    d = len(sample_marg_PDF_list)

    # initialization
    theta_list = []
    g_list     = []

    n_fail = 0
    for i in range(0, N):
        if (i % int(N/100) == 0):
            string = '[' + repr(int(i/N*100)) + '%] Iteration ' + repr(i)
            print(string)
        # initialization
        theta_star = np.zeros(d)

        # sample each dimension of itself
        for k in range(0, d):
            theta_star[k] = sample_marg_PDF_list[k]()

        # evaluate limit state function
        g_star = LSF(theta_star)

        # save theta and g in lists
        theta_list.append(theta_star)
        g_list.append(g_star)

    # calculate failure probability
    pf_mcs = sum(g<0 for g in g_list) / N

    return pf_mcs, theta_list, g_list
