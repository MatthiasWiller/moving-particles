"""
# ---------------------------------------------------------------------------
# File to produce plots for example 1 
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-07
# ---------------------------------------------------------------------------
# References:
# 1."MCMC algorithms for Subset Simulation"
#    Papaioannou, Betz, Zwirglmaier, Straub (2015)
# ---------------------------------------------------------------------------
"""

import time as timer

import numpy as np
import scipy.stats as scps
import matplotlib
import matplotlib.pyplot as plt

import utilities.plots as uplt
import utilities.util as uutil

matplotlib.rcParams.update({'font.size': 20})
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams['text.usetex'] = True

print("RUN 41_plot_example_1.py")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT
# ---------------------------------------------------------------------------

# parameters
d                   = 10        # number of dimensions

n_samples_per_level = 1000      # SUS: number of samples per conditional level
p0                  = 0.1       # SUS: Probability of each subset, chosen adaptively

n_initial_samples   = 100       # MP: Number of initial samples
b                   = 20        # MP: burnin

sigma_p_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
n_sigma = len(sigma_p_list)
# limit-state function
#beta = 5.1993       # for pf = 10^-7
# beta = 4.7534       # for pf = 10^-6
#beta = 4.2649       # for pf = 10^-5
#beta = 3.7190       # for pf = 10^-4
beta = 3.0902       # for pf = 10^-3
#beta = 2.3263       # for pf = 10^-2
LSF  = lambda u: u.sum(axis=0)/np.sqrt(d) + beta

# analytical CDF
analytical_CDF = lambda x: scps.norm.cdf(x, beta)

# ---------------------------------------------------------------------------
# LOAD RESULTS FROM SIMULATIONS
# ---------------------------------------------------------------------------
direction = 'python/data/sigma_p_study_sus/'

g_list_list_cs = []
g_list_list_mmh = []

iii = 0
for sigma_p in sigma_p_list:
    iii = iii + 1
    g_list_mp_tmp = np.load(direction + 'sus_example_1_d10_N1000_Nsim100_cs_sigmap' \
        + repr(iii) + '_g_list.npy')
    g_list_list_cs.append(g_list_mp_tmp)

iii = 0
for sigma_p in sigma_p_list:
    iii = iii + 1
    g_list_mp_tmp = np.load(direction + 'sus_example_1_d10_N1000_Nsim100_mmh_sigmap' \
        + repr(iii) + '_g_list.npy')
    g_list_list_mmh.append(g_list_mp_tmp)


# ---------------------------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------------------------

# initialization
pf_line_list  = []
b_line_list   = []
cov_line_list = []
legend_list   = []
cov_at_pf_array_cs = np.zeros(n_sigma, float)
pf_mean_array_cs = np.zeros(n_sigma, float)

cov_at_pf_array_mmh = np.zeros(n_sigma, float)
pf_mean_array_mmh = np.zeros(n_sigma, float)


#  conditional sampling
for i in range(0, n_sigma):
    b_line_sus, pf_line_list_sus        = uutil.get_pf_line_and_b_line_from_SUS(g_list_list_cs[i], p0, n_samples_per_level)
    pf_line_mean_sus, pf_line_cov_sus   = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_sus)

    b_line_list.append(b_line_sus)
    pf_line_list.append(pf_line_mean_sus)
    cov_line_list.append(pf_line_cov_sus)
    legend_list.append(r'SUS')
    cov_at_pf_array_cs[i] = pf_line_cov_sus[0]
    pf_mean_array_cs[i] = pf_line_mean_sus[0]


# modified metropolis hastings
for i in range(0, n_sigma):
    b_line_sus, pf_line_list_sus        = uutil.get_pf_line_and_b_line_from_SUS(g_list_list_mmh[i], p0, n_samples_per_level)
    pf_line_mean_sus, pf_line_cov_sus   = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_sus)

    b_line_list.append(b_line_sus)
    pf_line_list.append(pf_line_mean_sus)
    cov_line_list.append(pf_line_cov_sus)
    legend_list.append(r'SUS')
    cov_at_pf_array_mmh[i] = pf_line_cov_sus[0]
    pf_mean_array_mmh[i] = pf_line_mean_sus[0]

# analytical expression
pf_analytical  = np.ones(n_sigma, float) * analytical_CDF(0)
cov_analytical = np.ones(n_sigma, float) * np.sqrt(pf_analytical**(-1/n_initial_samples)-1)


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

# plot cov over standard deviation
plt.figure()

plt.plot(sigma_p_list, cov_analytical,'o', label=r'Analytical')
plt.plot(sigma_p_list, cov_at_pf_array_cs,'x', label=r'SuS with CS')
plt.plot(sigma_p_list, cov_at_pf_array_mmh,'+', label=r'SuS with MMH')



plt.legend()
plt.xlabel(r'Standard deviation, $\sigma_P$')
plt.ylabel(r'Coefficient of variation, $\hat{\delta}_{p_f}$')

plt.tight_layout()
plt.savefig('sigma_p_study_cov_over_b_sus.pdf', format='pdf', dpi=50, bbox_inches='tight')

# plot pf over d

plt.figure()
plt.plot(sigma_p_list, pf_analytical,'o', label=r'Analytical')
plt.plot(sigma_p_list, pf_mean_array_cs,'x', label=r'MP with CS')
plt.plot(sigma_p_list, pf_mean_array_mmh,'+', label=r'MP with MMH')

plt.yscale('log')

plt.legend()
plt.xlabel(r'Standard deviation, $\sigma_P$')
plt.ylabel(r'Probability of failure, $\hat{p}_f$')

plt.tight_layout()
plt.savefig('sigma_p_study_pf_over_b_sus.pdf', format='pdf', dpi=50, bbox_inches='tight')


plt.show()
