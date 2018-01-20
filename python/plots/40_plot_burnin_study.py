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
# Version 2017-10
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

savepdf = False
example = 2

# parameters
N = 500       # MP: Number of initial samples 

burn_in_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# burn_in_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,\
#                 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

# ---------------------------------------------------------------------------
# EXAMPLE INPUT
# ---------------------------------------------------------------------------

if example == 1:
    d = 10
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
    pf_ref = 5e-9

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
# direction = 'python/data/example' + repr(example) + '/burnin_study/'
# direction = 'python/data/example' + repr(example) + '/burnin_study_seed1/'
direction = 'python/data/example' + repr(example) + '/burnin_study_b-4/'



g_list_list = []

for Nb in burn_in_list:
    g_list_mp_tmp = np.load(direction + 'mp_' + example_name +'_N'+repr(N)+'_Nsim100_b'+ repr(Nb) +'_cs_sss2_g_list.npy')
    g_list_list.append(g_list_mp_tmp)

# ---------------------------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------------------------

# initialization
cov_at_pf_array = np.zeros(len(burn_in_list), float)
pf_mean_array = np.zeros(len(burn_in_list), float)

for i in range(0, len(burn_in_list)):
    pf_mean_array[i], cov_at_pf_array[i] = uutil.get_mean_and_cov_pf_from_MP(g_list_list[i], N)

# analytical expression
pf_ref  = np.ones(len(burn_in_list), float) * pf_ref
cov_ref = np.ones(len(burn_in_list), float) * np.sqrt(pf_ref**(-1/N)-1)

# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
# plot cov over b
plt.figure()
plt.plot(burn_in_list, cov_ref,'-', label=r'MP analytical')
plt.plot(burn_in_list, cov_at_pf_array,'+-', label=r'MP Strategy 2')

plt.legend()
plt.xlabel(r'Burn-in, $N_b$')
plt.ylabel(r'Coefficient of variation, $\hat{\delta}_{p_f}$')

plt.tight_layout()
if savepdf:
    plt.savefig('burnin_study_cov_over_Nb.pdf', format='pdf', dpi=50, bbox_inches='tight')


# plot pf over b
plt.figure()
plt.plot(burn_in_list, pf_ref,'-', label=r'Reference (MCS)')
plt.plot(burn_in_list, pf_mean_array,'v-', label=r'MP Strategy 2')

# plt.yticks([])


plt.yscale('log')

# if example == 2:
    # plt.ylim([2e-3, 3e-3])

if example == 3:
    plt.ylim([2e-5, 4e-5])

if example == 4:
    plt.ylim([1e-3, 1e-2])

# ax = plt.gca()
# ax.set_yscale('log')
# ax.set_ylim([1e-3, 1e-2])
# ax.set_yticks([0.01, 0.001])
# ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# yticks = ax.yaxis.get_major_ticks()
# for i in range(1, len(yticks)-1):
#     print(i)
#     yticks[i].label1.set_visible(False)


plt.legend(loc='lower left')
plt.xlabel(r'Burn-in, $N_b$')
plt.ylabel(r'Probability of failure, $\hat{p}_f$')

plt.tight_layout()
if savepdf:
    plt.savefig('burnin_study_pf_over_Nb.pdf', format='pdf', dpi=50, bbox_inches='tight')

plt.show()
