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

savepdf = True
example = 4

# parameters

# burn_in_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
burn_in_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# burn_in_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,\
#                 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

# ---------------------------------------------------------------------------
# EXAMPLE INPUT
# ---------------------------------------------------------------------------

if example == 1:
    d = 10
    pf_ref = 1e-3

    nsamples_list = [632, 337, 230, 175, 141, 118, 101, 89, 79, 71, 65, 60, 55, 51, 48, 45, 42, 40, 38, 36]
    

    # example_name
    example_name = 'example_1_d10'


elif example == 2:
     # reference solution from paper/MCS
    pf_ref = 2.275e-3

    nsamples_list = [706, 380, 260, 197, 159, 133, 115, 101, 90, 81, 74, 68, 62, 58, 54, 51, 48, 45, 43, 41]

    # example_name
    example_name = 'waarts'

elif example == 3:
    # reference solution from paper/MCS
    pf_ref = 3.17e-5

    # example_name
    example_name = 'breitung'

    nsamples_list = [440, 230, 156, 118, 95, 79, 68, 60, 53, 48, 43, 40, 37, 34, 32, 30, 28, 27, 25, 24]

elif example == 4:
    # reference solution from paper/MCS
    pf_ref = 4.42e-3

    # example_name
    example_name = 'liebscher'

    nsamples_list = [779, 422, 290, 220, 178, 149, 128, 113, 100, 91, 82, 76, 70, 65, 61, 57, 54, 51, 48, 46]

# ---------------------------------------------------------------------------
# LOAD RESULTS FROM SIMULATIONS
# ---------------------------------------------------------------------------
pf_mean_list = []
ncall_mean_list = []
# direction = 'python/data/example' + repr(example) + '/burnin_study/'
# direction = 'python/data/example' + repr(example) + '/burnin_study_seed1/'
direction = 'python/data/example' + repr(example) + \
            '/burnin_tradeoff_5e3/'

g_list_list = []


for i in range(0, len(burn_in_list)):
    N = nsamples_list[i]
    Nb = burn_in_list[i]
    g_list_mp_tmp = np.load(direction + 'mp_' + example_name +'_N'+repr(N)+'_Nsim100_b'+ repr(Nb) +'_cs_sss2_g_list.npy')
    g_list_list.append(g_list_mp_tmp)

# ---------------------------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------------------------

# initialization
cov_at_pf_array = np.zeros(len(burn_in_list), float)
pf_mean_array   = np.zeros(len(burn_in_list), float)
ncall_array     = np.zeros(len(burn_in_list), float)
cov_ref         = np.zeros(len(burn_in_list), float)

for i in range(0, len(burn_in_list)):
    pf_mean_array[i], cov_at_pf_array[i] = uutil.get_mean_and_cov_pf_from_MP(g_list_list[i], nsamples_list[i])
    ncall_array[i] = uutil.get_mean_ncall_from_MP(g_list_list[i], nsamples_list[i], burn_in_list[i])

    cov_ref[i] = np.sqrt(pf_ref**(-1/nsamples_list[i])-1)

pf_mean_list.append(pf_mean_array)
ncall_mean_list.append(ncall_array)
    
# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
# plot cov over b
plt.figure()

plt.plot(burn_in_list, cov_ref,'-', color='C0', label=r'analytical')
plt.plot(burn_in_list, cov_at_pf_array,'+-', label=r'$\bar{N}_{call} = 5 \cdot 10^3$', color='C1')

plt.legend()
plt.xlabel(r'Burn-in, $N_b$')
plt.ylabel(r'Coefficient of variation, $\hat{\delta}_{p_f}$')

plt.tight_layout()
if savepdf:
    plt.savefig('burnin_tradeoff_cov_over_Nb.pdf', format='pdf', dpi=50, bbox_inches='tight')


# plot pf over b
plt.figure()
pf_ref = np.ones(len(burn_in_list), float) * pf_ref
plt.plot(burn_in_list, pf_ref,'-', label=r'Reference (MCS)')

plt.plot(burn_in_list, pf_mean_array,'v-', label=r'$\bar{N}_{call} = 5 \cdot 10^3$', color='C1')

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
    plt.savefig('burnin_tradeoff_pf_over_Nb.pdf', format='pdf', dpi=50, bbox_inches='tight')


# plot Ncall
plt.figure()
for ncall_array in ncall_mean_list:
    plt.plot(burn_in_list, ncall_array,'-', label=r'MP')

plt.xlabel(r'Burn-in, $N_b$')
plt.ylabel(r'Number of calls to the LSF, $N_{call}$')
plt.tight_layout()

plt.show()
