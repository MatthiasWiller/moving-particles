"""
# ---------------------------------------------------------------------------
# Moving Particles Test Approach
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-06
# ---------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scps

import algorithms.mp_guyader_sampler as mpgs
import algorithms.mp_cond_sampler as mpcs

print("RUN 10_mp_test_approach.py")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT FOR MOVING PARTICLES
# ---------------------------------------------------------------------------

# parameters
N = 1000          # number of samples
d = 10           # number of dimensions

b     = 20
#sampler = mpgs.GuyaderSampler(b, 0.3)
sampler = mpcs.CondSampler(b, 0.8)

m_max = 1e7

# limit-state function
beta = 4.7534       # for pf = 10^-6
LSF  = lambda u: u.sum(axis=0)/np.sqrt(d) + beta

# analytical CDF
analytical_CDF = lambda x: scps.norm.cdf(x, beta)

# ---------------------------------------------------------------------------
# INPUT FOR MONTE CARLO SIMULATION (LEVEL 0)
# ---------------------------------------------------------------------------

# initialization
sample_marg_PDF_list = []
f_marg_PDF_list      = []

# sample from marginal pdf (gaussian)
sample_marg_PDF = lambda: np.random.randn(1)

# marginal pdf / target pdf (gaussian)
f_marg_PDF      = lambda x: np.exp(-0.5 * x**2)/np.sqrt(2*np.pi)

# append distributions to list
for i in range(0, d):
    sample_marg_PDF_list.append(sample_marg_PDF)
    f_marg_PDF_list.append(f_marg_PDF)


# ---------------------------------------------------------------------------
# MOVING PARTICLES
# ---------------------------------------------------------------------------


# initialization
theta = np.zeros((N, d), float)
g     = np.zeros(N, float)
acc   = 0


# MCS sampling
for i in range(0, N):
    for k in range(0, d):
        theta[i, k] = sample_marg_PDF_list[k]()

    g[i] = LSF(theta[i, :])

m = 0

while np.max(g) > 0 and m < m_max:
    # get index of smallest g
    id_min = np.argmax(g)

    # sampling
    theta_temp, g_temp = sampler.get_next_sample(theta[id_min], g[id_min], LSF)

    # count acceptance rate
    if g[id_min] != g_temp:
        acc = acc + 1

    theta[id_min] = theta_temp
    g[id_min]     = g_temp

    m = m + 1
    print('m:', m, ' | g =', g_temp)

pf_hat = (1 - 1/N)**m
# pf_hat = np.exp(-1/N)**m

# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------
pf_analytical    = analytical_CDF(0)

print("\nRESULTS:")
print("> Probability of Failure (Moving Particels Est.) =", round(pf_hat, 8))
print("> Probability of Failure (Analytical) \t\t=", round(pf_analytical, 8))
print("> Acceptance rate \t\t\t\t=", round(acc/m, 8))