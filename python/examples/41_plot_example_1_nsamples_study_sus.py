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

import numpy as np
import scipy.stats as scps
import matplotlib
import matplotlib.pyplot as plt

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

plotpdf = True

# parameters
d  = 10        # number of dimensions

p0 = 0.1       # SUS: Probability of each subset, chosen adaptively

# nsamples_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90, 100]

nsamples_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]


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
direction = 'python/data/example1/nsamples_study_sus/'

g_list_list = []

for N in nsamples_list:
    g_list_mp_tmp = np.load(direction + 'sus_example_1_d' + repr(d) + '_N'+ repr(N) +'_Nsim100_cs_g_list.npy')
    g_list_list.append(g_list_mp_tmp)


# ---------------------------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------------------------

# initialization
cov_at_pf_array = np.zeros(len(nsamples_list), float)
pf_mean_array   = np.zeros(len(nsamples_list), float)

for i in range(0, len(nsamples_list)):
    N = nsamples_list[i]
    b_line_sus, pf_line_list_sus        = uutil.get_pf_line_and_b_line_from_SUS(g_list_list[i], p0, N)
    pf_line_mean_sus, pf_line_cov_sus   = uutil.get_mean_and_cov_from_pf_lines(pf_line_list_sus)

    cov_at_pf_array[i] = pf_line_cov_sus[0]
    pf_mean_array[i]   = pf_line_mean_sus[0]

# analytical expression
pf_analytical = np.ones(len(nsamples_list), float) * analytical_CDF(0)
cov_analytical = np.zeros(len(nsamples_list), float)
for i in range(0,len(nsamples_list)):
    cov_analytical[i] = np.sqrt(pf_analytical[i]**(-1/nsamples_list[i])-1)

# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
# plot cov over b
plt.figure()
plt.plot(nsamples_list, cov_at_pf_array,'+-', label=r'SuS', color='C1')

plt.legend()
plt.xlabel(r'Number of samples, $N$')
plt.ylabel(r'Coefficient of variation, $\hat{\delta}_{p_f}$')

plt.tight_layout()
if plotpdf:
    plt.savefig('nsamples_study_cov_over_b_sus.pdf', format='pdf', dpi=50, bbox_inches='tight')


# plot pf over b
plt.figure()
plt.plot(nsamples_list, pf_analytical,'.-', label=r'analytical', color='C0')
plt.plot(nsamples_list, pf_mean_array,'+-', label=r'SuS', color='C1')


plt.yscale('log')

plt.legend()
plt.xlabel(r'Number of samples, $N$')
plt.ylabel(r'Probability of failure, $\hat{p}_f$')

plt.tight_layout()
if plotpdf:
    plt.savefig('nsamples_study_pf_over_b_sus.pdf', format='pdf', dpi=50, bbox_inches='tight')

plt.show()
