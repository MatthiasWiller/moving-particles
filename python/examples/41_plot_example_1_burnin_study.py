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
import matplotlib.pyplot as plt

import utilities.plots as uplt
import utilities.util as uutil

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
b_max               = 30        # MP: max number of burnins

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
direction = 'python/data/'

g_list_list = []

for i in range(1, b_max):
    g_list_mp_tmp = np.load(direction + 'mp_example_1_d10_N100_Nsim10_b'+ repr(i) +'_cs_wSeedSel_g_list.npy')
    g_list_list.append(g_list_mp_tmp)


# ---------------------------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------------------------

# initialization
pf_line_list  = []
b_line_list   = []
cov_line_list = []
legend_list   = []
mean_cov_array = np.zeros(29, float)

for i in range(0, b_max-1):
    b_line_mp, pf_line_list_mp        = uutil.get_pf_line_and_b_line_from_MP(g_list_list[i], n_initial_samples)
    pf_line_mean_mp, pf_line_cov_mp   = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_mp)

    b_line_list.append(b_line_mp)
    pf_line_list.append(pf_line_mean_mp)
    cov_line_list.append(pf_line_cov_mp)
    legend_list.append(r'MP (T =' + repr(i+1))
    mean_cov_array[i] = np.mean(pf_line_cov_mp)
    


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
plt.figure()
burn_in_list = [i for i in range(1, b_max)]
plt.plot(burn_in_list, mean_cov_array)

# uplt.plot_pf_over_b(b_line_list, pf_line_list, legend_list)
# uplt.plot_cov_over_b(b_line_list, cov_line_list, legend_list)

plt.show()
