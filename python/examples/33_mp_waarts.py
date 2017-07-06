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
# Version 2017-07
# ---------------------------------------------------------------------------
"""

import numpy as np

import algorithms.modified_metropolis as mmh
import algorithms.cond_sampling as cs

import algorithms.moving_particles as mp


print("RUN 33_mp_waarts.py")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT FOR MOVING PARTICLES
# ---------------------------------------------------------------------------

# parameters
N = 100                     # number of samples
b = 30                      # burn-in
sampling_method  = 'mmh'    # 'mmh' = Modified Metropolis Hastings
                            # 'cs'  = Conditional Sampling
n_simulations = 2           # number of simulations

# file-name
filename = 'python/data/mp_waarts_N' + repr(N) + '_Nsim' + repr(n_simulations) + '_b' + repr(b) + '_' + sampling_method

# reference value
pf_analytical = 2.275 * 10**-3


# limit-state function
LSF = lambda u: np.minimum(3 + 0.1*(u[0] - u[1])**2 - 2**(-0.5) * np.absolute(u[0] + u[1]), 7* 2**(-0.5) - np.absolute(u[0] - u[1]))


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
sample_marg_PDF_list.append(sample_marg_PDF)
sample_marg_PDF_list.append(sample_marg_PDF)
f_marg_PDF_list.append(f_marg_PDF)
f_marg_PDF_list.append(f_marg_PDF)


# ---------------------------------------------------------------------------
# MOVING PARTICLES
# ---------------------------------------------------------------------------

# initializing sampling method
if sampling_method == 'mmh':
    sampler = mmh.ModifiedMetropolisHastings(sample_marg_PDF_list, f_marg_PDF_list, 'gaussian', b)
elif sampling_method == 'cs':
    sampler = cs.CondSampling(sample_marg_PDF_list, 0.8, b)

# initialization
pf_list    = []
theta_list = []
g_list     = []

for sim in range(0, n_simulations):
    pf_hat, theta_temp, g_list, acc_rate, m_list = mp.mp_one_particle(N, LSF, sampler, sample_marg_PDF_list)
    # save simulation in list
    pf_list.append(pf_hat)
    g_list.append(g_temp)
    theta_list.append(theta_temp)

pf_sim_array = np.asarray(pf_list)
pf_mean      = np.mean(pf_sim_array)
pf_sigma     = np.std(pf_sim_array)

# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------


print("\nRESULTS:")
print("> Probability of Failure (Moving Particels Est.)\t=", round(pf_mean, 8))
print("> Probability of Failure (Analytical) \t\t\t=", round(pf_analytical, 8))
print("> Pf mean \t=", pf_mean)
print("> Pf sigma \t=", pf_sigma)
print("> C.O.V. \t=", pf_sigma/pf_mean)


# ---------------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------------

np.save(filename + '_g_list.npy', g_list)
np.save(filename + '_theta_list.npy', theta_list)
print('\n> File was successfully saved as:', filename)
