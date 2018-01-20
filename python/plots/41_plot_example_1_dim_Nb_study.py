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
# Version 2017-10
# ---------------------------------------------------------------------------
"""

import time as timer

import numpy as np
import scipy.stats as scps
import matplotlib
import matplotlib.pyplot as plt

import utilities.plots as uplt
import utilities.util as uutil

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams['text.usetex'] = True

print("RUN file")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT
# ---------------------------------------------------------------------------

savepdf = True

# parameters
d                   = 100        # number of dimensions

n_samples_per_level = 1000      # SUS: number of samples per conditional level
p0                  = 0.1       # SUS: Probability of each subset, chosen adaptively

n_initial_samples   = 100       # MP: Number of initial samples 

burn_in_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# burn_in_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

dim_list = [10, 100, 1000]


# limit-state function
#beta = 5.1993       # for pf = 10^-7
#beta = 4.7534       # for pf = 10^-6
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
direction = 'python/data/example1/'

g_list_list_d10   = []
g_list_list_d100  = []
g_list_list_d1000 = []


for Nb in burn_in_list:
    g_list_mp_tmp = np.load(direction + 'burnin_study_d10/mp_example_1_d10_N100_Nsim100_b'+ repr(Nb) +'_cs_sss2_g_list.npy')
    g_list_list_d10.append(g_list_mp_tmp)

for Nb in burn_in_list:
    g_list_mp_tmp = np.load(direction + 'burnin_study_d100/mp_example_1_d100_N100_Nsim100_b'+ repr(Nb) +'_cs_sss2_g_list.npy')
    g_list_list_d100.append(g_list_mp_tmp)

for Nb in burn_in_list:
    g_list_mp_tmp = np.load(direction + 'burnin_study_d1000/mp_example_1_d1000_N100_Nsim100_b'+ repr(Nb) +'_cs_sss2_g_list.npy')
    g_list_list_d1000.append(g_list_mp_tmp)


# ---------------------------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------------------------

# initialization
cov_at_pf_array_d10 = np.zeros(len(burn_in_list), float)
pf_mean_array_d10 = np.zeros(len(burn_in_list), float)

cov_at_pf_array_d100 = np.zeros(len(burn_in_list), float)
pf_mean_array_d100 = np.zeros(len(burn_in_list), float)

cov_at_pf_array_d1000 = np.zeros(len(burn_in_list), float)
pf_mean_array_d1000 = np.zeros(len(burn_in_list), float)


# d=10
for i in range(0, len(burn_in_list)):
    b_line_mp, pf_line_list_mp        = uutil.get_pf_line_and_b_line_from_MP(g_list_list_d10[i], n_initial_samples)
    pf_line_mean_mp, pf_line_cov_mp   = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_mp)

    cov_at_pf_array_d10[i] = pf_line_cov_mp[0]
    pf_mean_array_d10[i] = pf_line_mean_mp[0]

# d=100
for i in range(0, len(burn_in_list)):
    b_line_mp, pf_line_list_mp        = uutil.get_pf_line_and_b_line_from_MP(g_list_list_d100[i], n_initial_samples)
    pf_line_mean_mp, pf_line_cov_mp   = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_mp)

    cov_at_pf_array_d100[i] = pf_line_cov_mp[0]
    pf_mean_array_d100[i] = pf_line_mean_mp[0]

# d=1000
for i in range(0, len(burn_in_list)):
    b_line_mp, pf_line_list_mp        = uutil.get_pf_line_and_b_line_from_MP(g_list_list_d1000[i], n_initial_samples)
    pf_line_mean_mp, pf_line_cov_mp   = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_mp)

    cov_at_pf_array_d1000[i] = pf_line_cov_mp[0]
    pf_mean_array_d1000[i] = pf_line_mean_mp[0]

# analytical expression
pf_analytical = np.ones(len(burn_in_list), float) * analytical_CDF(0)
cov_analytical = np.ones(len(burn_in_list), float) * np.sqrt(pf_analytical**(-1/n_initial_samples)-1)

# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
# plot cov over b
plt.figure()
plt.plot(burn_in_list, cov_analytical,'-', label=r'Analytical')
plt.plot(burn_in_list, cov_at_pf_array_d10,'+-', label=r'd=10')
plt.plot(burn_in_list, cov_at_pf_array_d100,'x-', label=r'd=100')
plt.plot(burn_in_list, cov_at_pf_array_d1000,'v-', label=r'd=1000')


plt.legend()
plt.xlabel(r'Burn-in, $N_b$')
plt.ylabel(r'Coefficient of variation, $\hat{\delta}_{p_f}$')

plt.tight_layout()
if savepdf:
    plt.savefig('burnin_study_cov_over_Nb_dim.pdf', format='pdf', dpi=50, bbox_inches='tight')


# plot pf over b
plt.figure()
plt.plot(burn_in_list, pf_analytical,'-', label=r'Analytical')
plt.plot(burn_in_list, pf_mean_array_d10,'+-', label=r'd=10')
plt.plot(burn_in_list, pf_mean_array_d100,'x-', label=r'd=100')
plt.plot(burn_in_list, pf_mean_array_d1000,'v-', label=r'd=1000')


plt.yscale('log')

plt.legend()
plt.xlabel(r'Burn-in, $N_b$')
plt.ylabel(r'Probability of failure, $\hat{p}_f$')

plt.tight_layout()
if savepdf:
    plt.savefig('burnin_study_pf_over_Nb_dim.pdf', format='pdf', dpi=50, bbox_inches='tight')

plt.show()
