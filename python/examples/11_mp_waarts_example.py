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

import algorithms.mp_moving_particles as mp

import algorithms.mp_guyader_sampler as mpgs
import algorithms.mp_cond_sampler as mpcs
import algorithms.mp_mh_sampler as mpmhs
import algorithms.mp_mmh_sampler as mpmmhs

print("RUN 11_mp_waarts_example.py")

# set seed for randomization
# np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT FOR MOVING PARTICLES
# ---------------------------------------------------------------------------

# parameters
N = 100          # number of samples
d = 2            # number of dimensions
b = 30           # burn-in

n_simulations = 5

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
# MOVING PARTICLES
# ---------------------------------------------------------------------------

#sampler = mpgs.GuyaderSampler(b, 0.3)
#sampler = mpcs.CondSampler(b, 0.8)
#sampler = mpmhs.MHSampler(b, 0.3, f_marg_PDF_list)
sampler = mpmmhs.MMHSampler(b, 0.3, f_marg_PDF_list)

pf_list = []
for sim in range(0, n_simulations):
    pf_hat, theta_temp, g_temp, acc_rate = mp.mp_one_particle(N, LSF, sampler, sample_marg_PDF_list)
    # save simulation in list
    pf_list.append(pf_hat)

pf_sim_array = np.asarray(pf_list)
pf_mean = np.mean(pf_sim_array)
pf_sigma = np.std(pf_sim_array)

# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------

pf_analytical = 2.275 * 10**-3

print("\nRESULTS:")
print("> Probability of Failure (Moving Particels Est.)\t=", round(pf_mean, 8))
print("> Probability of Failure (Analytical) \t\t\t=", round(pf_analytical, 8))
print("> Pf mean =", pf_mean)
print("> Pf sigma =", pf_sigma)
print("> C.O.V. =", pf_sigma/pf_mean)