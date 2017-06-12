"""
# ---------------------------------------------------------------------------
# Subset Simulation Method example: Example 4.2 Ref. [2]
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-05
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
import matplotlib.pyplot as plt
import scipy.stats as scps

import algorithms.sus as sus
import algorithms.cond_sampling as cs
import algorithms.modified_metropolis as mmh
import algorithms.adaptive_cond_sampling as acs

import utilities.plots as uplt
import utilities.stats as ustat
import utilities.util as uutil

print("RUN 07_sus_example_2.py")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT FOR SUBSET SIMULATION
# ---------------------------------------------------------------------------

# parameters
n_samples_per_level = 500          # number of samples per conditional level
p0                  = 0.1           # Probability of each subset, chosen adaptively

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

# analytical CDF
# no analytical CDF available

# ---------------------------------------------------------------------------
# INPUT FOR MONTE CARLO SIMULATION (LEVEL 0)
# ---------------------------------------------------------------------------

# initialization
sample_marg_PDF_list = []
f_marg_PDF_list      = []



# sample from marginal pdf (beta-distribution)
#sample_marg_PDF = lambda: scps.beta.rvs(p, q, 1)
#sample_marg_PDF = lambda: np.random.beta(p, q, 1)
#sample_marg_PDF = lambda: transform_X2U(scps.beta.rvs(p, q, size=1))
sample_marg_PDF = lambda: transform_X2U(beta_distr.rvs(1))

# marginal pdf / target pdf (beta-distribution)
#f_marg_PDF      = lambda x: scps.beta.pdf(x, p, q)
#f_marg_PDF      = lambda u: scps.beta.pdf(transform_U2X(u), p, q)
f_marg_PDF      = lambda u: beta_distr.pdf(transform_U2X(u))

# append distributions to list
sample_marg_PDF_list.append(sample_marg_PDF)
sample_marg_PDF_list.append(sample_marg_PDF)
f_marg_PDF_list.append(f_marg_PDF)
f_marg_PDF_list.append(f_marg_PDF)


# ---------------------------------------------------------------------------
# INPUT FOR MODIFIED METROPOLIS HASTINGS
# ---------------------------------------------------------------------------
# distributions
mu      = 0.0
sigma   = 2.0

# proposal distribution (gaussian)
f_prop_PDF      = lambda x, param: ( 2.0*np.pi*sigma**2.0 )**-.5 * np.exp( -.5 * (x - param)**2. / sigma**2. )

# sample from proposal distribution (gaussian)
sample_prop_PDF = lambda param: np.random.normal(param, sigma, 1)


# ---------------------------------------------------------------------------
# INPUT FOR CONDITIONAL SAMPLING
# ---------------------------------------------------------------------------

# sample from conditional PDF
sample_cond_PDF = lambda mu_cond, sigma_cond: np.random.normal(mu_cond, sigma_cond, 1)

# note: don't set it to 0.2; it is too low;
rho_k = 0.8         # ~0.7 gives kinda good results

# ---------------------------------------------------------------------------
# INPUT FOR ADAPTIVE CONDITIONAL SAMPLING
# ---------------------------------------------------------------------------

# sample from conditional PDF
sample_cond_PDF = lambda mu_cond, sigma_cond: np.random.normal(mu_cond, sigma_cond, 1)

#
pa = 0.1

# ---------------------------------------------------------------------------
# SUBSET SIMULATION
# ---------------------------------------------------------------------------

# initializing sampling method
#sampling_method = mmh.ModifiedMetropolisHastings(sample_marg_PDF_list, f_marg_PDF_list, sample_prop_PDF, f_prop_PDF)
#sampling_method = cs.CondSampling(sample_marg_PDF_list, sample_cond_PDF, rho_k)
sampling_method = acs.AdaptiveCondSampling(sample_marg_PDF_list, sample_cond_PDF, pa)


# apply subset-simulation
n_sim = 1

# initialization of lists
p_F_SS_list  = []
theta_list   = []
g_list       = []


print('\n> START Sampling')
startTime = timer.time()

n_loops = n_sim
while n_loops > 0:
    for i in range(0, n_loops):
        # perform SubSim
        p_F_SS, theta, g = sus.subsetsim(p0, n_samples_per_level, LSF, sampling_method)

        # transform samples from u to x-space
        for j in range(0, len(theta)):
            theta[j] = transform_U2X(theta[j])

        # save values in lists
        p_F_SS_list.append(p_F_SS)
        theta_list.append(theta)
        g_list.append(g)
        print("> [", i+1, "] Subset Simulation Estimator \t=", p_F_SS)

    # check if we have enough samples yet
    n_eff_sim = uutil.get_n_eff_sim(g_list)
    n_loops = n_sim - n_eff_sim

print("\n> Time needed for Sampling =", round(timer.time() - startTime, 2), "s")

# computing cov
print('\n> START Computing C.O.V')
startTime = timer.time()
delta     = ustat.cov_analytical(theta, g, p0, n_samples_per_level, p_F_SS)
print("> Time needed for Computing C.O.V =", round(timer.time() - startTime, 2), "s")

# ---------------------------------------------------------------------------
# RESULTS
# --------------------------------------------------------------------------

p_F_SS_array    = np.asarray(p_F_SS_list).reshape(-1)
sigma_pf_ss     = np.std(p_F_SS_array)
mu_pf_ss        = np.mean(p_F_SS_array)

mu_pf_mcs       = 0.00405

delta_analytical = delta
delta_estimation = sigma_pf_ss/mu_pf_ss

print("\nSTART Results:")
print("> Probability of Failure (SubSim Est.)\t=", round(mu_pf_ss, 8))
print("> Probability of Failure (MCS) \t\t=", round(mu_pf_mcs, 8))
print("> Coefficient of Variation (Estimation)\t=", round(delta_estimation, 8))
print("> Coefficient of Variation (Analytical)\t=", round(delta_analytical, 8))


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

# plot samples
uplt.plot_sus_list(g_list, p0, n_samples_per_level, p_F_SS_array, analytical_CDF=0)
#plt.show()
g_max_global = np.amax(np.asarray(g).reshape(-1))
for i in range(0, len(theta)):
    uplt.plot_surface_with_samples(theta[i], g[i], z, g_max_global)

plt.show()
