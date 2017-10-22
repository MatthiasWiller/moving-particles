"""
# ---------------------------------------------------------------------------
# Test function to test plots
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-09
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

savepdf = False

example_list = [1, 2, 3, 4, 5]

# ---------------------------------------------------------------------------
# DEFINE FUNCTIONS
# ---------------------------------------------------------------------------
p0 = 0.1

# analytical CDF for example 1
d              = 10     
beta           = 3.0902       # for pf = 10^-3
analytical_CDF = lambda x: scps.norm.cdf(x, beta)
pf_ref_1       = analytical_CDF(0)

example_name_list = ['example_1_d10', 'waarts',   'breitung', 'liebscher', 'au_beck']
pf_ref_list       = [pf_ref_1,        2.275e-3,   3.17e-5,    4.42e-3,     6e-7     ]
N_sus_list        = [1000,            420,        5000,       10000,       1000     ]
N_mp_list         = [104,             40,         478,        1105,        100      ]


# initialization
pf_rel_line_sus = np.zeros(len(example_list))
pf_rel_line_mp  = np.zeros(len(example_list))
cov_line_sus    = np.zeros(len(example_list))
cov_line_mp     = np.zeros(len(example_list))


# ---------------------------------------------------------------------------
# LOAD RESULTS FROM SIMULATIONS
# ---------------------------------------------------------------------------
# -- load sus data ----------------------------------------------------------
for i in range(0, len(example_list)):
    example = example_list[i]
    direction = 'python/data/example' + repr(example_list[i]) + '/fixed_ncall_data/'
    
    N = N_sus_list[i]
    g_list_list_sus = \
        np.load(direction + 'sus_' + example_name_list[i] + '_N' + repr(N) + '_Nsim100_cs_g_list.npy')

    pf_temp, cov_temp = \
        uutil.get_mean_and_cov_pf_from_SUS(g_list_list_sus, N, p0)

    pf_rel_line_sus[i] = pf_temp/pf_ref_list[i]
    cov_line_sus[i]    = cov_temp
    

# -- load mp data ----------------------------------------------------------
for i in range(0, len(example_list)):
    example = example_list[i]
    direction = 'python/data/example' + repr(example_list[i]) + '/fixed_ncall_data/'
    
    N = N_mp_list[i]
    g_list_list_mp = \
        np.load(direction + 'mp_' + example_name_list[i] + '_N' + repr(N) + '_Nsim100_b5_cs_sss2_g_list.npy')

    pf_temp, cov_temp = \
        uutil.get_mean_and_cov_pf_from_MP(g_list_list_mp, N)

    pf_rel_line_mp[i] = pf_temp/pf_ref_list[i]
    cov_line_mp[i]    = cov_temp


# analytical reference
ref_lines = np.ones(len(example_list))

# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------


plt.figure()
plt.plot(example_list, ref_lines, '-', color='C0')

plt.plot(example_list, pf_rel_line_mp, 'o', color='C1', label='MP')
plt.plot(example_list, pf_rel_line_sus, 's', color='C2', label='SuS')

# legend
plt.legend()

# xaxis
plt.xlabel(r'Example')
plt.xticks(example_list)


# yaxis
plt.ylabel(r'Rel. probabilitly of failure, $\frac{\hat{p}_f}{p_f^{MCS}}$')
plt.ylim(0.95, 1.05)

plt.tight_layout()
if savepdf:
    plt.savefig('pf_rel_over_example.pdf', format='pdf', dpi=50, bbox_inches='tight')


plt.figure()
plt.plot(example_list, cov_line_mp, 'o', color='C1', label='MP')
plt.plot(example_list, cov_line_sus, 's', color='C2', label='SuS')

# legend
plt.legend()

# xaxis
plt.xlabel(r'Example')
plt.xticks(example_list)

# yaxis
plt.ylabel(r'Coefficient of variation, $\hat{\delta}_{P_f}$')

plt.tight_layout()
if savepdf:
    plt.savefig('cov_over_example.pdf', format='pdf', dpi=50, bbox_inches='tight')

plt.show()