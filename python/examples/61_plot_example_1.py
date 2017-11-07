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

print("RUN file")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT
# ---------------------------------------------------------------------------

savepdf = False

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
direction = 'python/data/example1/fixed_ncall_data/'

g_list_sus     = np.load(direction + 'sus_example_1_d10_N1000_Nsim100_cs_g_list.npy')

g_list_mp1     = np.load(direction + 'mp_example_1_d10_N104_Nsim100_b5_cs_sss2_g_list.npy')
g_list_mp2     = np.load(direction + 'mp_example_1_d10_N53_Nsim100_b10_cs_sss2_g_list.npy')
g_list_mp3     = np.load(direction + 'mp_example_1_d10_N27_Nsim100_b20_cs_sss2_g_list.npy')


# ---------------------------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------------------------

print('Ncall,SUS =', uutil.get_mean_ncall_from_SUS(g_list_sus, 1000, 0.1))
print('Ncall,MP1 =', uutil.get_mean_ncall_from_MP(g_list_mp1, 104, 5))
print('Ncall,MP2 =', uutil.get_mean_ncall_from_MP(g_list_mp2, 53, 10))
print('Ncall,MP3 =', uutil.get_mean_ncall_from_MP(g_list_mp3, 27, 20))

pf_mean_sus, pf_cov_sus = uutil.get_mean_and_cov_pf_from_SUS(g_list_sus, 1000, 0.1)
pf_mean_mp1, pf_cov_mp1 = uutil.get_mean_and_cov_pf_from_MP(g_list_mp1, 104)
pf_mean_mp2, pf_cov_mp2 = uutil.get_mean_and_cov_pf_from_MP(g_list_mp2, 52)
pf_mean_mp3, pf_cov_mp3 = uutil.get_mean_and_cov_pf_from_MP(g_list_mp3, 27)

print('SUS: pf =', pf_mean_sus, '| cov =', pf_cov_sus)
print('MP1: pf =', pf_mean_mp1, '| cov =', pf_cov_mp1)
print('MP2: pf =', pf_mean_mp2, '| cov =', pf_cov_mp2)
print('MP3: pf =', pf_mean_mp3, '| cov =', pf_cov_mp3)

b_line_analytical       = np.linspace(0,5,100)
pf_line_analytical      = analytical_CDF(b_line_analytical)

b_line_sus, pf_line_list_sus      = uutil.get_pf_line_and_b_line_from_SUS(g_list_sus, 0.1, 1000)
pf_line_mean_sus, pf_line_cov_sus = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_sus)

b_line_mp1, pf_line_list_mp1        = uutil.get_pf_line_and_b_line_from_MP(g_list_mp1, 104)
pf_line_mean_mp1, pf_line_cov_mp1   = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_mp1)

b_line_mp2, pf_line_list_mp2      = uutil.get_pf_line_and_b_line_from_MP(g_list_mp2, 52)
pf_line_mean_mp2, pf_line_cov_mp2 = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_mp2)

b_line_mp3, pf_line_list_mp3      = uutil.get_pf_line_and_b_line_from_MP(g_list_mp3, 27)
pf_line_mean_mp3, pf_line_cov_mp3 = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_mp3)

# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

# -- Pf over b ----------------------------------
plt.figure()

plt.plot(b_line_analytical, pf_line_analytical, '-', label='analytical', color='C0')
plt.plot(b_line_sus, pf_line_mean_sus, '--', label='SuS $N=1000$', color='C1')
plt.plot(b_line_mp1, pf_line_mean_mp1, '--', label='MP $N=104$, $N_b = 5$', color='C2')
plt.plot(b_line_mp2, pf_line_mean_mp2, '--', label='MP $N=53$, $N_b = 10$', color='C3')
plt.plot(b_line_mp3, pf_line_mean_mp3, '--', label='MP $N=27$, $N_b = 20$', color='C4')

# set y-axis to log-scale
plt.yscale('log')

# add legend
matplotlib.rcParams['legend.fontsize'] = 14
plt.legend(loc='lower right')

# set labels
plt.xlabel(r'Limit state function values $b$')
plt.ylabel(r'$P(g(x) \leq b)$')
plt.tight_layout()

if savepdf:
    plt.savefig('plot_pf_over_b.pdf', format='pdf', dpi=50, bbox_inches='tight')



# -- cov over b ----------------------------------
plt.figure()

plt.plot(b_line_sus, pf_line_cov_sus, '--', label='SuS $N=1000$', color='C1')
plt.plot(b_line_mp1, pf_line_cov_mp1, '--', label='MP $N=104$, $N_b = 5$', color='C2')
plt.plot(b_line_mp2, pf_line_cov_mp2, '--', label='MP $N=53$, $N_b = 10$', color='C3')
plt.plot(b_line_mp3, pf_line_cov_mp3, '--', label='MP $N=27$, $N_b = 20$', color='C4')

# add legend
matplotlib.rcParams['legend.fontsize'] = 14
plt.legend(loc='upper right')

# set labels
plt.xlabel(r'Limit state function values, $b$')
plt.ylabel(r'Coefficient of variation, $\hat{\delta}_{P_f}$')
plt.tight_layout()

if savepdf:
    plt.savefig('plot_cov_over_b.pdf', format='pdf', dpi=50, bbox_inches='tight')

plt.show()