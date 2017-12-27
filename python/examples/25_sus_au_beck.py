"""
# ---------------------------------------------------------------------------
# Subset Simulation Method Example 5
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
# 1."MCMC algorithms for Subset Simulation"
#    Papaioannou, Betz, Zwirglmaier, Straub (2015)
# 2. Efficiency Improvement of Stochastic Simulation by Means of Subset Sampling
#    Martin Liebscher, Stephan Pannier, Jan-Uwe Sickert, Wolfgang Graf (2006)
# ---------------------------------------------------------------------------
"""

import time as timer
import numpy as np

import algorithms.modified_metropolis as mmh
import algorithms.cond_sampling as cs
import algorithms.adaptive_cond_sampling as acs

import algorithms.sus as sus

import SDOF as sdof

import utilities.stats as ustat
import utilities.util as uutil

print("RUN file")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# DEFINITION OF THE WHITE NOISE ESCITATION
# ---------------------------------------------------------------------------

S  = 1                          # White noise spectral intensity 
T  = 30                         # Duration of the excitation, s
dt = 0.02                       # Time increment, s
t  = np.arange(0,T+2*dt,dt)     # time instants (one more due to interpolation)
n  = len(t)-1                   # n points ~ number of random variables
# The uncertain state vector theta consists of the sequence of i.i.d.
# standard Gaussian random variables which generate the white noise input
# at the discrete time instants
W = lambda theta: np.sqrt(2*np.pi*S/dt)*theta   # random excitation


# ---------------------------------------------------------------------------
# STANDARD INPUT FOR SUBSET SIMULATION
# ---------------------------------------------------------------------------

# parameters
n_samples_per_level = 1000          # number of samples per conditional level
p0                  = 0.1          # Probability of each subset, chosen adaptively
sampling_method     = 'cs'         # 'mmh' = Modified Metropolis Hastings
                                    # 'cs'  = Conditional Sampling
                                    # 'acs' = adaptive Conditional Sampling
n_simulations       = 100             # Number of Simulations


# file-name
direction = 'python/data/'
filename = direction + 'sus_au_beck_N' + repr(n_samples_per_level) + '_Nsim' + repr(n_simulations) + '_' + sampling_method


# limit-state function
max_thresh = 2.4    # See Fig.(1) Ref.(1)
lsf = lambda theta: sdof.LSF(theta, t, W, max_thresh)

# reference solution from MCS
pf_mcs = 0

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
for i in range(0, n+1):
    sample_marg_PDF_list.append(sample_marg_PDF)
    f_marg_PDF_list.append(f_marg_PDF)

# ---------------------------------------------------------------------------
# SUBSET SIMULATION
# ---------------------------------------------------------------------------

# initializing sampling method
if sampling_method == 'mmh':
     sampler = mmh.ModifiedMetropolisHastings(sample_marg_PDF_list, f_marg_PDF_list, 'gaussian')
elif sampling_method == 'cs':
    sampler = cs.CondSampling(sample_marg_PDF_list, 0.8)
elif sampling_method == 'acs':
    sampler = acs.AdaptiveCondSampling(sample_marg_PDF_list, 0.1)


## apply subset-simulation

# initialization of lists
p_F_SS_list  = []
theta_list   = []
g_list       = []


print('\n> START Sampling')
start_time = timer.time()
for sim in range(0, n_simulations):
    # perform SubSim
    p_F_SS, theta, g = sus.subsetsim(p0, n_samples_per_level, lsf, sampler)

    # save values in lists
    p_F_SS_list.append(p_F_SS)
    theta_list.append(theta)
    g_list.append(g)

    uutil.print_simulation_progress(sim, n_simulations, start_time)


print("\n> Time needed for Sampling =", round(timer.time() - start_time, 2), "s")

# computing cov
print('\n> START Computing C.O.V')
startTime = timer.time()

cov_analytical = ustat.cov_analytical(theta, g, p0, n_samples_per_level, p_F_SS)

print("> Time needed for Computing C.O.V =", round(timer.time() - startTime, 2), "s")

# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------

p_F_SS_array   = np.asarray(p_F_SS_list).reshape(-1)
sigma_pf_ss    = np.std(p_F_SS_array)
mu_pf_ss       = np.mean(p_F_SS_array)

cov_estimation = sigma_pf_ss/mu_pf_ss

print("\nSTART Results:")
print("> Probability of Failure (SubSim Est.)\t=", round(mu_pf_ss, 8))
print("> Probability of Failure (Monte Carlo) \t\t=", round(pf_mcs, 8))
print("> Coefficient of Variation (Estimation)\t=", round(cov_estimation, 8))
print("> Coefficient of Variation (Analytical)\t=", round(cov_analytical, 8))


# ---------------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------------

np.save(filename + '_g_list.npy', g_list)
# np.save(filename + '_theta_list.npy', theta_list)
print("\n> File was successfully saved as:", filename)
