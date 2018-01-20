"""
# ---------------------------------------------------------------------------
# Plot Computational Time over pf/cov for example 1
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

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import scipy.stats as scps

import utilities.util as uutil

matplotlib.rcParams.update({'font.size': 20})
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams['text.usetex'] = True
   
# ---------------------------------------------------------------------------
# STANDARD INPUT
# ---------------------------------------------------------------------------

# parameters
d                   = 10        # number of dimensions

n_samples_per_level = 1000      # SUS: number of samples per conditional level
p0                  = 0.1       # SUS: Probability of each subset, chosen adaptively

n_initial_samples   = 100       # MP: Number of initial samples 
Nb                  = 5        # MP: number of burnins

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
g_list_list_mp = np.load(direction + 'mp_example_1_d10_N100_Nsim2_b5_cs_sss2_g_list.npy')


b_line_mp, pf_line_list_mp = \
    uutil.get_pf_line_and_b_line_from_MP(g_list_list_mp, n_initial_samples)

pf_line_mp, cov_line_mp = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_mp)

pf_line_mp, ncall_line_list_mp = \
    uutil.get_ncall_lines_and_pf_line_from_MP(b_line_mp, pf_line_mp, g_list_list_mp, n_initial_samples, Nb)


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------


plt.figure()
plt.plot(pf_line_mp, ncall_line_list_mp[0])

# xaxis
plt.xscale('log')
plt.gca().invert_xaxis()
plt.xlabel(r'Probability of failure, $p_f$')

# yaxis
plt.yscale('log')
# plt.ylim(0, 5e13)
plt.ylabel(r'Computational cost, $N_{call}$')

plt.show()