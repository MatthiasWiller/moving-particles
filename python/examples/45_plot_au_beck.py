"""
# ---------------------------------------------------------------------------
# File to produce plots
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
import matplotlib.pyplot as plt

import utilities.plots as uplt
import utilities.util as uutil

print("RUN 44_plot_au_beck.py")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT
# ---------------------------------------------------------------------------

# parameters
n_samples_per_level = 1000      # SUS: number of samples per conditional level
p0                  = 0.1       # SUS: Probability of each subset, chosen adaptively

n_initial_samples   = 100       # MP: Number of initial samples 


# ---------------------------------------------------------------------------
# LOAD RESULTS FROM SIMULATIONS
# ---------------------------------------------------------------------------
direction = 'python/data/example5/fixed_ncall_data/'

mcs_data = np.loadtxt(direction + 'mcs_au_beck_N10000000.txt', delimiter=',')
# g_list_mcs = mcs_data[:,1]
b_line_mcs = mcs_data[:,0]
pf_line_mcs = mcs_data[:,1]
# g_list_mcs = np.load(direction + 'mcs_au_beck_N200_g_list.npy')

# g_list_sus = np.load(direction + 'sus_example_1_d10_Nspl1000_Nsim2_acs_g_list.npy')

# g_list_mp = np.load(direction + 'mp_example_1_d10_N100_Nsim2_b30_mmh_g_list.npy')

# ---------------------------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------------------------

#b_line_mcs, pf_line_mcs         = uutil.get_pf_line_and_b_line_from_MCS(g_list_mcs)

# b_line_list_sus, pf_line_sus    = uutil.get_pf_line_and_b_line_from_SUS(g_list_sus, p0, n_samples_per_level)

# b_line_list_mp, pf_line_list_mp = uutil.get_pf_line_and_b_line_from_MP(g_list_mp, n_initial_samples)


# initialization
b_line_list  = []
pf_line_list = []
legend_list  = []

b_line_list.append(b_line_mcs)
# b_line_list.append(b_line_list_sus[0])
# b_line_list.append(b_line_list_mp[0])

pf_line_list.append(pf_line_mcs)
# pf_line_list.append(pf_line_sus)
# pf_line_list.append(pf_line_list_mp[0])

legend_list.append('MCS')
# legend_list.append('SUS')
# legend_list.append('MP')


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

# plot samples

uplt.plot_pf_over_b(b_line_list, pf_line_list, legend_list)

plt.show()

# ---------------------------------------------------------------------------
# PLOTS (from orinial file - doesn't work here!!)
# ---------------------------------------------------------------------------

# plot samples
# g_max_global = np.amax(np.asarray(g).reshape(-1))
# for i in range(0, len(theta)):
#     uplt.plot_surface_with_samples(theta[i], g[i], z, g_max_global)

# plt.show()