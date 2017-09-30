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
# Version 2017-09
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

from matplotlib.pyplot import cm
from matplotlib import rcParams

# create figure object with LaTeX font
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 22
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

import utilities.plots as uplt
import utilities.util as uutil

print("RUN 41_plot_example_1.py")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT
# ---------------------------------------------------------------------------

savepdf = False

p0 = 0.1
Nb = 5
# cov = [0.1, 0.2, 0.3]
N_mp  = [100, 200, 100]   # seed = (0, 0, 0)
N_sus = [6000, 1500, 750] # seed = (0, 1, 3)


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
direction = 'python/data/example1/table_data/'

g_list_sus1 = np.load(direction + 'sus_example_1_d10_N' + repr(N_sus[0]) + '_Nsim100_cs_g_list.npy')
g_list_sus2 = np.load(direction + 'sus_example_1_d10_N' + repr(N_sus[1]) + '_Nsim100_cs_g_list.npy')
g_list_sus3 = np.load(direction + 'sus_example_1_d10_N' + repr(N_sus[2]) + '_Nsim100_cs_g_list.npy')

g_list_mp1 = np.load(direction + 'mp_example_1_d10_N' + repr(N_mp[0]) + '_Nsim100_b5_cs_sss2_g_list.npy')
g_list_mp2 = np.load(direction + 'mp_example_1_d10_N' + repr(N_mp[1]) + '_Nsim100_b5_cs_sss2_g_list.npy')
g_list_mp3 = np.load(direction + 'mp_example_1_d10_N' + repr(N_mp[2]) + '_Nsim100_b5_cs_sss2_g_list.npy')


# ---------------------------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------------------------

# coefficient of variation
pf_mean_sus1, pf_cov_sus1 = uutil.get_mean_and_cov_pf_from_SUS(g_list_sus1, N_sus[0], p0)
pf_mean_sus2, pf_cov_sus2 = uutil.get_mean_and_cov_pf_from_SUS(g_list_sus2, N_sus[1], p0)
pf_mean_sus3, pf_cov_sus3 = uutil.get_mean_and_cov_pf_from_SUS(g_list_sus3, N_sus[2], p0)

pf_mean_mp1, pf_cov_mp1 = uutil.get_mean_and_cov_pf_from_MP(g_list_mp1, N_mp[0])
pf_mean_mp2, pf_cov_mp2 = uutil.get_mean_and_cov_pf_from_MP(g_list_mp2, N_mp[1])
pf_mean_mp3, pf_cov_mp3 = uutil.get_mean_and_cov_pf_from_MP(g_list_mp3, N_mp[2])
print('-----------------------------------------------------')
print('cov = 0.1 | SUS cov =', round(pf_cov_sus1, 3), '| MP cov =', round(pf_cov_mp1, 3))
print('cov = 0.2 | SUS cov =', round(pf_cov_sus2, 3), '| MP cov =', round(pf_cov_mp2, 3))
print('cov = 0.3 | SUS cov =', round(pf_cov_sus3, 3), '| MP cov =', round(pf_cov_mp3, 3))
print('-----------------------------------------------------')

# number of LSF calls
ncall_sus1 = uutil.get_mean_ncall_from_SUS(g_list_sus1, N_sus[0], p0)
ncall_sus2 = uutil.get_mean_ncall_from_SUS(g_list_sus2, N_sus[1], p0)
ncall_sus3 = uutil.get_mean_ncall_from_SUS(g_list_sus3, N_sus[2], p0)

ncall_mp1 = uutil.get_mean_ncall_from_MP(g_list_mp1, N_mp[0], Nb)
ncall_mp2 = uutil.get_mean_ncall_from_MP(g_list_mp2, N_mp[1], Nb)
ncall_mp3 = uutil.get_mean_ncall_from_MP(g_list_mp3, N_mp[2], Nb)

print('cov = 0.1 | SUS ncall =', round(ncall_sus1, 0), '| MP ncall =', round(ncall_mp1, 0))
print('cov = 0.2 | SUS ncall =', round(ncall_sus2, 0), '| MP ncall =', round(ncall_mp2, 0))
print('cov = 0.3 | SUS ncall =', round(ncall_sus3, 0), '| MP ncall =', round(ncall_mp3, 0))
print('-----------------------------------------------------')
print('Thank you for the music!')