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

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams['text.usetex'] = True

print("RUN 41_plot_example_1.py")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT
# ---------------------------------------------------------------------------

savepdf = False

# parameters
d                   = 100        # number of dimensions

n_samples_per_level = 1000      # SUS: number of samples per conditional level
p0                  = 0.1       # SUS: Probability of each subset, chosen adaptively

n_initial_samples   = 100       # MP: Number of initial samples 
b_max               = 10        # MP: max number of burnins

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
direction = 'python/data/burnin_study_d100/'

g_list_list_sss0 = []
g_list_list_sss1 = []
g_list_list_sss2 = []
g_list_list_sss3 = []
g_list_list_sss4 = []
g_list_list_sss5 = []


# for i in range(0, b_max):
#     g_list_mp_tmp = np.load(direction + 'mp_example_1_d'+ repr(d) +'_N100_Nsim50_b'+ repr(i+1) +'_cs_sss0_g_list.npy')
#     g_list_list_sss0.append(g_list_mp_tmp)

for i in range(0, b_max):
    g_list_mp_tmp = np.load(direction + 'mp_example_1_d'+ repr(d) +'_N100_Nsim50_b'+ repr(i+1) +'_cs_sss1_g_list.npy')
    g_list_list_sss1.append(g_list_mp_tmp)

for i in range(0, b_max):
    g_list_mp_tmp = np.load(direction + 'mp_example_1_d'+ repr(d) +'_N100_Nsim50_b'+ repr(i+1) +'_cs_sss2_g_list.npy')
    g_list_list_sss2.append(g_list_mp_tmp)

for i in range(0, b_max):
    g_list_mp_tmp = np.load(direction + 'mp_example_1_d'+ repr(d) +'_N100_Nsim50_b'+ repr(i+1) +'_cs_sss3_g_list.npy')
    g_list_list_sss3.append(g_list_mp_tmp)

for i in range(0, b_max):
    g_list_mp_tmp = np.load(direction + 'mp_example_1_d'+ repr(d) +'_N100_Nsim50_b'+ repr(i+1) +'_cs_sss4_g_list.npy')
    g_list_list_sss4.append(g_list_mp_tmp)

for i in range(0, b_max):
    g_list_mp_tmp = np.load(direction + 'mp_example_1_d'+ repr(d) +'_N100_Nsim50_b'+ repr(i+1) +'_cs_sss5_g_list.npy')
    g_list_list_sss5.append(g_list_mp_tmp)


# ---------------------------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------------------------

# initialization
cov_at_pf_array_sss0 = np.zeros(b_max, float)
pf_mean_array_sss0 = np.zeros(b_max, float)

cov_at_pf_array_sss1 = np.zeros(b_max, float)
pf_mean_array_sss1 = np.zeros(b_max, float)

cov_at_pf_array_sss2 = np.zeros(b_max, float)
pf_mean_array_sss2 = np.zeros(b_max, float)

cov_at_pf_array_sss3 = np.zeros(b_max, float)
pf_mean_array_sss3 = np.zeros(b_max, float)

cov_at_pf_array_sss4 = np.zeros(b_max, float)
pf_mean_array_sss4 = np.zeros(b_max, float)

cov_at_pf_array_sss5 = np.zeros(b_max, float)
pf_mean_array_sss5 = np.zeros(b_max, float)

# seed selection stragegy 0
# for i in range(0, b_max):
#     b_line_mp, pf_line_list_mp        = uutil.get_pf_line_and_b_line_from_MP(g_list_list_sss0[i], n_initial_samples)
#     pf_line_mean_mp, pf_line_cov_mp   = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_mp)

#     cov_at_pf_array_sss0[i] = pf_line_cov_mp[0]
#     pf_mean_array_sss0[i] = pf_line_mean_mp[0]

# seed selection strategy 1
for i in range(0, b_max):
    b_line_mp, pf_line_list_mp        = uutil.get_pf_line_and_b_line_from_MP(g_list_list_sss1[i], n_initial_samples)
    pf_line_mean_mp, pf_line_cov_mp   = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_mp)

    cov_at_pf_array_sss1[i] = pf_line_cov_mp[0]
    pf_mean_array_sss1[i] = pf_line_mean_mp[0]

# seed selection strategy 2
for i in range(0, b_max):
    b_line_mp, pf_line_list_mp        = uutil.get_pf_line_and_b_line_from_MP(g_list_list_sss2[i], n_initial_samples)
    pf_line_mean_mp, pf_line_cov_mp   = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_mp)

    cov_at_pf_array_sss2[i] = pf_line_cov_mp[0]
    pf_mean_array_sss2[i] = pf_line_mean_mp[0]

# seed selection strategy 3
for i in range(0, b_max):
    b_line_mp, pf_line_list_mp        = uutil.get_pf_line_and_b_line_from_MP(g_list_list_sss3[i], n_initial_samples)
    pf_line_mean_mp, pf_line_cov_mp   = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_mp)

    cov_at_pf_array_sss3[i] = pf_line_cov_mp[0]
    pf_mean_array_sss3[i] = pf_line_mean_mp[0]

# seed selection strategy 4
for i in range(0, b_max):
    b_line_mp, pf_line_list_mp        = uutil.get_pf_line_and_b_line_from_MP(g_list_list_sss4[i], n_initial_samples)
    pf_line_mean_mp, pf_line_cov_mp   = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_mp)

    cov_at_pf_array_sss4[i] = pf_line_cov_mp[0]
    pf_mean_array_sss4[i] = pf_line_mean_mp[0]

# seed selection strategy 5
for i in range(0, b_max):
    b_line_mp, pf_line_list_mp        = uutil.get_pf_line_and_b_line_from_MP(g_list_list_sss5[i], n_initial_samples)
    pf_line_mean_mp, pf_line_cov_mp   = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_mp)

    cov_at_pf_array_sss5[i] = pf_line_cov_mp[0]
    pf_mean_array_sss5[i] = pf_line_mean_mp[0]

# analytical expression
pf_analytical = np.ones(b_max, float) * analytical_CDF(0)
cov_analytical = np.ones(b_max, float) * np.sqrt(pf_analytical**(-1/n_initial_samples)-1)

# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
# plot cov over b
plt.figure()
burn_in_list = [i for i in range(1, b_max+1)]
plt.plot(burn_in_list, cov_analytical,'.-', label=r'Analytical')
# plt.plot(burn_in_list, cov_at_pf_array_sss0,'+-', label=r'Strategy 0')
plt.plot(burn_in_list, cov_at_pf_array_sss1,'x-', label=r'Strategy 1')
plt.plot(burn_in_list, cov_at_pf_array_sss2,'v-', label=r'Strategy 2')
# plt.plot(burn_in_list, cov_at_pf_array_sss3,'*-', label=r'Strategy 3')
# plt.plot(burn_in_list, cov_at_pf_array_sss4,'d-', label=r'Strategy 4')
plt.plot(burn_in_list, cov_at_pf_array_sss5,'s-', label=r'Strategy 5')

plt.legend()
plt.xlabel(r'Burn-in, $N_b$')
plt.ylabel(r'Coefficient of variation, $\hat{\delta}_{p_f}$')

plt.tight_layout()
if savepdf:
    plt.savefig('burnin_study_cov_over_Nb_d1000.pdf', format='pdf', dpi=50, bbox_inches='tight')


# plot pf over b
plt.figure()
plt.plot(burn_in_list, pf_analytical,'.-', label=r'Analytical')
# plt.plot(burn_in_list, pf_mean_array_sss0,'+-', label=r'Strategy 0')
plt.plot(burn_in_list, pf_mean_array_sss1,'x-', label=r'Strategy 1')
plt.plot(burn_in_list, pf_mean_array_sss2,'v-', label=r'Strategy 2')
# plt.plot(burn_in_list, pf_mean_array_sss3,'*-', label=r'Strategy 3')
# plt.plot(burn_in_list, pf_mean_array_sss4,'d-', label=r'Strategy 4')
plt.plot(burn_in_list, pf_mean_array_sss5,'s-', label=r'Strategy 5')

plt.yscale('log')

plt.legend()
plt.xlabel(r'Burn-in, $N_b$')
plt.ylabel(r'Probability of failure, $\hat{p}_f$')

plt.tight_layout()
if savepdf:
    plt.savefig('burnin_study_pf_over_Nb_d1000.pdf', format='pdf', dpi=50, bbox_inches='tight')

plt.show()
