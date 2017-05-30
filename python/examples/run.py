"""
Author: Matthias Willer 2017
"""

import time as timer

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scps

import algorithms.sus as sus
import algorithms.cond_sampling as cs
import algorithms.modified_metropolis as mmh
import algorithms.adaptive_cond_sampling as acs

import plots.sus_plot as splt
import plots.user_plot as uplt

np.random.seed(0)

n_samples = 25000

# parameters for beta-distribution
p = 6.0
q = 6.0
beta_distr = scps.beta(p, q)

# limit-state function
z   = lambda x: 8* np.exp(-(x[0]**2 + x[1]**2)) + 2* np.exp(-((x[0]-5)**2 + (x[1]-4)**2)) + 1 + x[0]*x[1]/10
LSF = lambda x: 7.5 - z(x)

sample_marg_PDF = lambda: beta_distr.rvs(1)



theta0  = np.zeros((n_samples, 2), float)
g0      = np.zeros(n_samples, float)

Nf = 0

for i in range(0, n_samples):
    msg = "> MCS ... [" + repr(int(i/n_samples*100)) + "%]"
    print(msg)

    # sample theta0
    for k in range(0, 2):
        theta0[i, k] = sample_marg_PDF()

    # evaluate theta0
    g0[i] = LSF(theta0[i, :])

    # count failure samples
    if g0[i] <= 0:
        Nf += 1

pf_mcs = Nf/n_samples

print(pf_mcs)