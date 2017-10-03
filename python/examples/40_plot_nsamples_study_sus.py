"""
# ---------------------------------------------------------------------------
# File to produce plots for examples 1-4
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

print("RUN file")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT
# ---------------------------------------------------------------------------

plotpdf = True
example = 3

# parameters
p0 = 0.1       # SUS: Probability of each subset, chosen adaptively

# nsamples_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90, 100]
nsamples_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]

# ---------------------------------------------------------------------------
# EXAMPLE INPUT
# ---------------------------------------------------------------------------


if example == 1:
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
    pf_ref = analytical_CDF(0)

    # example_name
    example_name = 'example_1_d10'


elif example == 2:
    # limit-state function
    LSF = lambda u: np.minimum(3 + 0.1*(u[0] - u[1])**2 - 2**(-0.5) * np.absolute(u[0] + u[1]), 7* 2**(-0.5) - np.absolute(u[0] - u[1]))
    
    # reference solution from paper/MCS
    pf_ref = 2.275e-3

    # example_name
    example_name = 'waarts'


elif example == 3:
    # limit-state function
    # LSF = lambda x: np.minimum(5-x[0], 4+x[1])
    LSF = lambda x: np.minimum(5-x[0], 1/(1+np.exp(-2*(x[1]+4)))-0.5)

    # reference solution from paper/MCS
    pf_ref = 3.17e-5

    # example_name
    example_name = 'breitung'


elif example == 4:
    # limit-state function
    z   = lambda x: 8* np.exp(-(x[0]**2 + x[1]**2)) + 2* np.exp(-((x[0]-5)**2 + (x[1]-4)**2)) + 1 + x[0]*x[1]/10
    LSF = lambda x: 7.5 - z(x)

    # reference solution from paper/MCS
    pf_ref = 4.42e-3

    # example_name
    example_name = 'liebscher'

# ---------------------------------------------------------------------------
# LOAD RESULTS FROM SIMULATIONS
# ---------------------------------------------------------------------------
direction = 'python/data/example' + repr(example) + '/nsamples_study_sus/'

g_list_list = []

for N in nsamples_list:
    g_list_mp_tmp = np.load(direction + 'sus_' + example_name + '_N'+ repr(N) +'_Nsim100_cs_g_list.npy')
    g_list_list.append(g_list_mp_tmp)


# ---------------------------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------------------------

# initialization
cov_at_pf_array = np.zeros(len(nsamples_list), float)
pf_mean_array   = np.zeros(len(nsamples_list), float)

for i in range(0, len(nsamples_list)):
    pf_mean_array[i], cov_at_pf_array[i] = \
        uutil.get_mean_and_cov_pf_from_SUS(g_list_list[i], nsamples_list[i], p0)

# analytical expression
pf_ref = np.ones(len(nsamples_list), float) * pf_ref
# cov_ref = np.zeros(len(nsamples_list), float)
# for i in range(0, len(nsamples_list)):
#     cov_ref[i] = np.sqrt(pf_ref[i]**(-1/nsamples_list[i])-1)

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
plt.plot(nsamples_list, pf_ref,'-', label=r'Reference (MCS)', color='C0')
plt.plot(nsamples_list, pf_mean_array,'+-', label=r'SuS', color='C1')


plt.yscale('log')
if example == 2:
    plt.ylim([2e-3, 3e-3])
if example == 3:
    plt.ylim([1e-5, 2e-4])

plt.legend()
plt.xlabel(r'Number of samples, $N$')
plt.ylabel(r'Probability of failure, $\hat{p}_f$')

plt.tight_layout()
if plotpdf:
    plt.savefig('nsamples_study_pf_over_b_sus.pdf', format='pdf', dpi=50, bbox_inches='tight')

plt.show()
