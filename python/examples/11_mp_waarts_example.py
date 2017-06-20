"""
# ---------------------------------------------------------------------------
# Moving Particles Waarts example
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
# np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT FOR MOVING PARTICLES
# ---------------------------------------------------------------------------

# parameters
N = 1000          # number of samples
d = 2            # number of dimensions

b     = 20
#sampler = mpgs.GuyaderSampler(b, 0.3)
sampler = mpcs.CondSampler(b, 0.8)

m_max = 1e7


# limit-state function
LSF = lambda u: min(3 + 0.1*(u[0] - u[1])**2 - 2**(-0.5) * np.absolute(u[0] + u[1]), 7* 2**(-0.5) - np.absolute(u[0] - u[1]))

# analytical CDF
# no analytical CDF available
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
# INPUT FOR CONDITIONAL SAMPLING
# ---------------------------------------------------------------------------

# sample from conditional PDF
sample_cond_PDF = lambda mu_cond, sigma_cond: np.random.normal(mu_cond, sigma_cond, 1)


# ---------------------------------------------------------------------------
# MOVING PARTICLES
# ---------------------------------------------------------------------------

# initialization
theta = np.zeros((N, d), float)
g = np.zeros(N, float)
acc = 0

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
    print('m:', m, '| g =', g_temp)

pf_hat = (1 - 1/N)**m
# pf_hat = np.exp(-1/N)**m

# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------
pf_analytical = 2.275 * 10**-3

print("\nRESULTS:")
print("> Probability of Failure (Moving Particels Est.)\t=", round(pf_hat, 8))
print("> Probability of Failure (Analytical) \t\t\t=", round(pf_analytical, 8))