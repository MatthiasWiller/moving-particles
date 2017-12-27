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

matplotlib.rcParams.update({'font.size': 23})
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams['text.usetex'] = True

print("RUN file")

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
d_max               = 20        # MP: max number of burnins

# limit-state function
#beta = 5.1993       # for pf = 10^-7
beta = 4.7534       # for pf = 10^-6
#beta = 4.2649       # for pf = 10^-5
#beta = 3.7190       # for pf = 10^-4
#beta = 3.0902       # for pf = 10^-3
#beta = 2.3263       # for pf = 10^-2
LSF  = lambda u: u.sum(axis=0)/np.sqrt(d) + beta

# analytical CDF
analytical_CDF = lambda x: scps.norm.cdf(x, beta)

# ---------------------------------------------------------------------------
# LOAD RESULTS FROM SIMULATIONS
# ---------------------------------------------------------------------------
direction = 'python/data/dimension_study/'

g_list_list_wSS = []
g_list_list_woSS = []

for i in range(1, d_max+1):
    g_list_mp_tmp = np.load(direction + 'mp_example_1_d' + repr(i) + '_N100_Nsim50_b20_cs_wSeedSel_g_list.npy')
    g_list_list_wSS.append(g_list_mp_tmp)

for i in range(1, d_max+1):
    g_list_mp_tmp = np.load(direction + 'mp_example_1_d' + repr(i) + '_N100_Nsim50_b20_cs_withoutSeedSel_g_list.npy')
    g_list_list_woSS.append(g_list_mp_tmp)

# ---------------------------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------------------------

# initialization
pf_line_list  = []
b_line_list   = []
cov_line_list = []
legend_list   = []
cov_at_pf_array_wSS = np.zeros(d_max, float)
pf_mean_array_wSS = np.zeros(d_max, float)

cov_at_pf_array_woSS = np.zeros(d_max, float)
pf_mean_array_woSS = np.zeros(d_max, float)


# without Seed selection
for i in range(0, d_max):
    b_line_mp, pf_line_list_mp        = uutil.get_pf_line_and_b_line_from_MP(g_list_list_woSS[i], n_initial_samples)
    pf_line_mean_mp, pf_line_cov_mp   = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_mp)

    b_line_list.append(b_line_mp)
    pf_line_list.append(pf_line_mean_mp)
    cov_line_list.append(pf_line_cov_mp)
    legend_list.append(r'MP (T =' + repr(i+1)+') w/o Seed Sel.')
    cov_at_pf_array_woSS[i] = pf_line_cov_mp[0]
    pf_mean_array_woSS[i] = pf_line_mean_mp[0]
    
# with Seed selection
for i in range(0, d_max):
    b_line_mp, pf_line_list_mp        = uutil.get_pf_line_and_b_line_from_MP(g_list_list_wSS[i], n_initial_samples)
    pf_line_mean_mp, pf_line_cov_mp   = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_mp)

    b_line_list.append(b_line_mp)
    pf_line_list.append(pf_line_mean_mp)
    cov_line_list.append(pf_line_cov_mp)
    legend_list.append(r'MP (T =' + repr(i+1)+') w/o Seed Sel.')
    cov_at_pf_array_wSS[i] = pf_line_cov_mp[0]
    pf_mean_array_wSS[i] = pf_line_mean_mp[0]

# analytical expression
pf_analytical  = np.ones(d_max, float) * analytical_CDF(0)
cov_analytical = np.ones(d_max, float) * np.sqrt(pf_analytical**(-1/n_initial_samples)-1)


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

# plot cov over b
plt.figure()
dim_list = [i for i in range(1, d_max+1)]
plt.plot(dim_list, cov_analytical,'o', label=r'Analytical')
plt.plot(dim_list, cov_at_pf_array_woSS,'x', label=r'without Seed Sel.')
plt.plot(dim_list, cov_at_pf_array_wSS,'+', label=r'without Seed Sel.')


plt.legend()
plt.xlabel(r'Dimension, $d$')
plt.ylabel(r'Coefficient of variation, $\hat{\delta}_{p_f}$')

plt.tight_layout()
plt.savefig('dimension_study_cov_over_b.pdf', format='pdf', dpi=50, bbox_inches='tight')



# plot pf over d

plt.figure()
plt.plot(dim_list, pf_analytical,'o', label=r'Analytical')
plt.plot(dim_list, pf_mean_array_woSS,'x', label=r'without Seed Sel.')
plt.plot(dim_list, pf_mean_array_wSS,'+', label=r'without Seed Sel.')

plt.yscale('log')

plt.legend()
plt.xlabel(r'Dimension, $d$')
plt.ylabel(r'Probability of failure, $\hat{p}_f$')

plt.tight_layout()
plt.savefig('dimension_study_pf_over_b.pdf', format='pdf', dpi=50, bbox_inches='tight')

# uplt.plot_pf_over_b(b_line_list, pf_line_list, legend_list)
# uplt.plot_cov_over_b(b_line_list, cov_line_list, legend_list)

plt.show()
