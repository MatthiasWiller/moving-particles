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

import plots.sus_plot as splt

print("RUN 07_sus_example_2.py")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT FOR SUBSET SIMULATION
# ---------------------------------------------------------------------------

# parameters
n_samples_per_level = 4000          # number of samples per conditional level
d                   = 2            # number of dimensions
p0                  = 0.1           # Probability of each subset, chosen adaptively

# limit-state function
LSF = lambda x: 7.5 - (8* np.exp(- (x[0]**2 + x[1]**2)) + 2* np.exp(-((x[0]-5)**2 + (x[1]-4)**2)) + 1 + x[0]*x[1]/10)


# ---------------------------------------------------------------------------
# INPUT FOR MODIFIED METROPOLIS HASTINGS
# ---------------------------------------------------------------------------
# distributions
mu      = 0.0
sigma   = 2.0
p       = 6.0
q       = 6.0

# marginal pdf / target pdf (beta-distribution)
#f_marg_PDF      = lambda x: np.exp(-0.5 * x**2)/np.sqrt(2*np.pi)
f_marg_PDF      = lambda x: scps.beta.pdf(x, p, q)


# sample from marginal pdf (beta-distribution)
#sample_marg_PDF = lambda dim: np.random.randn(dim[0], dim[1])
#sample_marg_PDF = lambda dim: scps.beta.rvs(p, q, dim)
sample_marg_PDF = lambda dim: np.random.beta(p, q, (dim[0], dim[1]))

# proposal distribution (uniform)
#f_prop_PDF      = lambda x, param: 0.5
#f_prop_PDF      = lambda x, param: np.exp(-0.5 * x**2)/np.sqrt(2*np.pi)
f_prop_PDF      = lambda x, param: ( 2.0*np.pi*sigma**2.0 )**-.5 * np.exp( -.5 * (x - param)**2. / sigma**2. )
#f_prop_PDF      = lambda x, param: scps.random.normal.pdf(x, param, sigma)

# sample from proposal distribution (uniform)
#sample_prop_PDF = lambda param: np.random.uniform(param-1, param+1, 1)
#sample_prop_PDF = lambda param: scps.norm.rvs(mu, sigma, 1)
sample_prop_PDF = lambda param: np.random.normal(param, sigma, 1)



# ---------------------------------------------------------------------------
# INPUT FOR CONDITIONAL SAMPLING
# ---------------------------------------------------------------------------

# sample from marginal pdf (exponential)
#sample_marg_PDF = lambda dim: np.random.randn(dim[0], dim[1])
#sample_marg_PDF = lambda dim: np.random.exponential(1/lam, (dim[0], dim[1]))
#sample_marg_PDF = lambda dim: scps.expon.rvs(0, lam, (dim[0], dim[1]))

# sample from conditional PDF
sample_cond_PDF = lambda mu_cond, sigma_cond: np.random.normal(mu_cond, sigma_cond, 1)

# note: don't set it to 0.2; it is too low; 
# 0.7 gives kinda good results
rho_k = 0.8

# ---------------------------------------------------------------------------
# INPUT FOR ADAPTIVE CONDITIONAL SAMPLING
# ---------------------------------------------------------------------------

# sample from marginal pdf (exponential)
#sample_marg_PDF = lambda dim: np.random.exponential(1/lam, (dim[0], dim[1]))

# sample from conditional PDF
sample_cond_PDF = lambda mu_cond, sigma_cond: np.random.normal(mu_cond, sigma_cond, 1)

pa = 0.1

# ---------------------------------------------------------------------------
# SUBSET SIMULATION
# ---------------------------------------------------------------------------

# initializing sampling method
sampling_method = mmh.ModifiedMetropolisHastings(sample_marg_PDF, f_marg_PDF, sample_prop_PDF, f_prop_PDF)
#sampling_method = cs.CondSampling(sample_marg_PDF, sample_cond_PDF, rho_k)
#sampling_method = acs.AdaptiveCondSampling(sample_marg_PDF, sample_cond_PDF, pa)


# apply subset-simulation
n_loops = 10
#p_F_SS_array = np.zeros(n_loops)
p_F_SS_list  = []
theta_list   = []
g_list       = []


print('\n> START Sampling')
startTime = timer.time()
for i in range(0, n_loops):
    p_F_SS, theta, g = sus.subsetsim(p0, n_samples_per_level, d, LSF, sampling_method)
    # p_F_SS_array[i] = p_F_SS

    # save values in lists
    p_F_SS_list.append(p_F_SS)
    theta_list.append(theta)
    g_list.append(g)

    print("> [", i+1, "] Subset Simulation Estimator \t=", p_F_SS)

print("\n> Time needed for Sampling =", round(timer.time() - startTime, 2), "s")

# computing cov
print('\n> START Computing C.O.V')
startTime = timer.time()
delta = sus.cov_analytical(theta, g, p0, n_samples_per_level, p_F_SS)
print("> Time needed for Computing C.O.V =", round(timer.time() - startTime, 2), "s")

# ---------------------------------------------------------------------------
# RESULTS
# --------------------------------------------------------------------------

# sigma_pf_ss = np.sqrt(np.var(p_F_SS_array))
# mu_pf_ss = np.mean(p_F_SS_array)

p_F_SS_array = np.asarray(p_F_SS_list).reshape(-1)
sigma_pf_ss  = np.std(p_F_SS_array)
mu_pf_ss     = np.mean(p_F_SS_array)
mu_pf_mcs    = 0.00405

print("\nSTART Results:")
print("> Probability of Failure (SubSim Est.)\t=", mu_pf_ss)
print("> Probability of Failure (Analytical) \t=", mu_pf_mcs)
print("> Coefficient of Variation (Estimation)\t=", sigma_pf_ss/mu_pf_ss)
print("> Coefficient of Variation (Analytical)\t=", round(delta, 8))


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

# analytical CDF
#analytical_CDF = lambda x: 1 - scps.gamma.cdf(x, d, lam)

# plot samples
splt.plot_sus(g, p0, n_samples_per_level, p_F_SS, analytical_CDF=0)
#splt.plot_sus_list(g_list, p0, n_samples_per_level, p_F_SS_array, analytical_CDF)
plt.show()
