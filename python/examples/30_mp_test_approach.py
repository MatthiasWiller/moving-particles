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

import algorithms.moving_particles as mp

import algorithms.modified_metropolis as mmh
import algorithms.cond_sampling as cs

import utilities.plots as uplt


print("RUN 30_mp_test_approach.py")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT FOR MOVING PARTICLES
# ---------------------------------------------------------------------------

# parameters
N = 100          # number of samples
d = 10           # number of dimensions
b = 30           # burn-in

n_simulations = 1

# limit-state function
beta = 3.0902       # for pf = 10^-3
LSF  = lambda u: u.sum(axis=0)/np.sqrt(d) + beta

# analytical CDF
analytical_CDF = lambda x: scps.norm.cdf(x, beta)

pf_analytical    = analytical_CDF(0)

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

#sampler = cs.CondSampling(sample_marg_PDF_list, 0.8, b)
sampler = mmh.ModifiedMetropolisHastings(sample_marg_PDF_list, f_marg_PDF_list, 'gaussian', b)

pf_list = []
for sim in range(0, n_simulations):
    pf_hat, theta_temp, g_list, acc_rate = mp.mp(N, LSF, sampler, sample_marg_PDF_list)
    # save simulation in list
    pf_list.append(pf_hat)

pf_sim_array = np.asarray(pf_list)
pf_mean      = np.mean(pf_sim_array)
pf_sigma     = np.std(pf_sim_array)

pf_analytical    = analytical_CDF(0)

# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------

print("\nRESULTS:")
print("> Probability of Failure (Moving Particels Est.) =", round(pf_mean, 8))
print("> Probability of Failure (Analytical) \t\t=", round(pf_analytical, 8))
print("> Acceptance rate \t\t\t\t=", round(acc_rate, 8))
print("> Pf mean =", pf_mean)
print("> Pf sigma =", pf_sigma)
print("> C.O.V. =", pf_sigma/pf_mean)

# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

uplt.plot_mp_pf(N, g_list, analytical_CDF)
plt.show()