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
import scipy.stats as scps

import algorithms.modified_metropolis as mmh
import algorithms.cond_sampling as cs

import algorithms.moving_particles as mp

print("RUN 32_mp_liebscher.py")

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
filename = 'python/data/mp_liebscher_N' + repr(N) + '_Nsim' + repr(n_simulations) + '_b' + repr(b) + '_' + sampling_method

# reference solution
mu_pf_mcs = 0.00405

# parameters for beta-distribution
p = 6.0
q = 6.0
beta_distr = scps.beta(p, q, loc=-2, scale=8)

# transformation to/from U-space
phi     = lambda x: scps.norm.cdf(x)
phi_inv = lambda x: scps.norm.ppf(x)

#CDF     = lambda x: scps.beta.cdf(x, p, q)
CDF     = lambda x: beta_distr.cdf(x)
#CDF_inv = lambda x: scps.beta.ppf(x, p, q)
CDF_inv = lambda x: beta_distr.ppf(x)

transform_U2X = lambda u: CDF_inv(phi(u))
transform_X2U = lambda x: phi_inv(CDF(x))

# limit-state function
z   = lambda x: 8* np.exp(-(x[0]**2 + x[1]**2)) + 2* np.exp(-((x[0]-5)**2 + (x[1]-4)**2)) + 1 + x[0]*x[1]/10
#LSF = lambda x: 7.5 - z(x)
LSF = lambda u: 7.5 - z(transform_U2X(u))
# ---------------------------------------------------------------------------
# INPUT FOR MONTE CARLO SIMULATION (LEVEL 0)
# ---------------------------------------------------------------------------

# initialization
sample_marg_PDF_list = []
f_marg_PDF_list      = []

# sample from marginal pdf (beta-distribution)
sample_marg_PDF = lambda: transform_X2U(beta_distr.rvs(1))

# marginal pdf / target pdf (beta-distribution)
f_marg_PDF      = lambda u: beta_distr.pdf(transform_U2X(u))

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

## apply moving particles - simulation

# initialization
pf_list    = []
theta_list = []
g_list     = []

for sim in range(0, n_simulations):
    pf_hat, theta_temp, g_temp, acc_rate, m_list = mp.mp_one_particle(N, LSF, sampler, sample_marg_PDF_list)
    
    # transform samples back from u to x-space
    for j in range(0, len(theta_temp)):
        theta_temp[j] = transform_U2X(theta_temp[j])

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
