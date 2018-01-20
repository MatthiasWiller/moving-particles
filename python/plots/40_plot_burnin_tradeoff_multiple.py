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
example = 1

ncall_number_list = [ 1,2,3]       # ncall_list = ['1e3', '5e3', '1e4', '5e4', '1e5']


# parameters

burn_in_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# burn_in_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# burn_in_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,\
#                 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

# ---------------------------------------------------------------------------
# EXAMPLE INPUT
# ---------------------------------------------------------------------------

d = 10

# pf_name = '1e-3'
# pf_name = '1e-6'
pf_name = '1e-9'
ncall_list = ['1e3', '5e3', '1e4', '5e4', '1e5']
ncall_label_list = [r'1 \cdot 10^3', r'5 \cdot 10^3', r'1 \cdot 10^4', r'5 \cdot 10^4', r'1 \cdot 10^5']


if pf_name == '1e-3':
    pf_ref = 1e-3
    nsamples_list_list = [[127, 68, 46, 35, 28, 24, 20, 18, 16, 14, 13, 12, 11, 10, 10, 9, 8, 8, 8, 7],\
                        [632, 337, 230, 175, 141, 118, 101, 89, 79, 71, 65, 60, 55, 51, 48, 45, 42, 40, 38, 36],\
                        [1265, 675, 460, 349, 281, 236, 203, 178, 158, 143, 130, 119, 110, 102, 96, 90, 84, 80, 76, 72],\
                        [6323, 3375, 2302, 1746, 1407, 1178, 1013, 889, 792, 713, 649, 596, 551, 512, 478, 448, 422, 399, 378, 359]]
elif pf_name == '1e-6':
    pf_ref = 1e-6
    nsamples_list_list = [[68, 35, 24, 18, 14, 12, 10, 9, 8, 7, 7, 6, 6, 5, 5, 5, 4, 4, 4, 4],\
                        [337, 175, 118, 89, 71, 60, 51, 45, 40, 36, 33, 30, 28, 26, 24, 23, 21, 20, 19, 18],\
                        [675, 349, 236, 178, 143, 119, 102, 90, 80, 72, 65, 60, 55, 51, 48, 45, 42, 40, 38, 36],\
                        [3375, 1746, 1178, 889, 713, 596, 512, 448, 399, 359, 327, 300, 277, 257, 240, 225, 212, 200, 190, 180],\
                        [6750, 3493, 2356, 1777, 1427, 1192, 1023, 897, 798, 719, 654, 600, 554, 514, 480, 450, 424, 401, 380, 361]]
elif pf_name == '1e-9':
    pf_ref = 1e-9
    nsamples_list_list = [[46, 24, 16, 12, 10, 8, 7, 6, 5, 5, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2],\
                        [230, 118, 79, 60, 48, 40, 34, 30, 27, 24, 22, 20, 18, 17, 16, 15, 14, 13, 13, 12],\
                        [460, 236, 158, 119, 96, 80, 69, 60, 53, 48, 44, 40, 37, 34, 32, 30, 28, 27, 25, 24],\
                        [2302, 1178, 792, 596, 478, 399, 342, 300, 267, 240, 218, 200, 185, 172, 160, 150, 142, 134, 127, 120],\
                        [4603, 2356, 1583, 1192, 956, 798, 685, 600, 533, 480, 437, 401, 370, 343, 321, 301, 283, 267, 253, 241]]

# example_name
example_name = 'example_1_d10'


# ---------------------------------------------------------------------------
# LOAD RESULTS FROM SIMULATIONS
# ---------------------------------------------------------------------------
cov_ref_list = []
cov_mean_list =  []
pf_mean_list = []
ncall_mean_list = []
# direction = 'python/data/example' + repr(example) + '/burnin_study/'
# direction = 'python/data/example' + repr(example) + '/burnin_study_seed1/'
for j in ncall_number_list:
    direction = 'python/data/example' + repr(example) + \
                '/burnin_tradeoff_pf' + pf_name + '/ncall_' + ncall_list[j] + '/'

    g_list_list = []
    nsamples_list = nsamples_list_list[j]

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

    cov_ref_list.append(cov_ref)
    cov_mean_list.append(cov_at_pf_array)
    pf_mean_list.append(pf_mean_array)
    ncall_mean_list.append(ncall_array)
    
# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
# plot cov over b
plt.figure()
for i in range(0, len(ncall_number_list)):
    ncall_nr = ncall_number_list[i]
    plt.plot(burn_in_list, cov_ref_list[i],'--', color='C' + repr(i+1))
    plt.plot(burn_in_list, cov_mean_list[i],'+-', label=r'$\bar{N}_{call} = '+ ncall_label_list[ncall_nr] + r'$', color='C' + repr(i+1))

plt.legend()
plt.xlabel(r'Burn-in, $N_b$')
plt.ylabel(r'Coefficient of variation, $\hat{\delta}_{p_f}$')
if pf_name == '1e-3':
    plt.ylim([0, 1.2])
elif pf_name == '1e-6':
    plt.ylim([0, 1.1])
elif pf_name == '1e-9':
    plt.ylim([0, 2.5])



plt.tight_layout()
if savepdf:
    plt.savefig('burnin_tradeoff_cov_over_Nb_'+pf_name+'.pdf', format='pdf', dpi=50, bbox_inches='tight')


# plot pf over b
plt.figure()
pf_ref = np.ones(len(burn_in_list), float) * pf_ref
plt.plot(burn_in_list, pf_ref,'-', label=r'Reference (MCS)')
for i in range(0, len(ncall_number_list)):
    ncall_nr = ncall_number_list[i]
    plt.plot(burn_in_list, pf_mean_list[i],'v-', label=r'$\bar{N}_{call} = '+ ncall_label_list[ncall_nr] + r'$', color='C' + repr(i+1))

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
    plt.savefig('burnin_tradeoff_pf_over_Nb_'+pf_name+'.pdf', format='pdf', dpi=50, bbox_inches='tight')


# plot Ncall
plt.figure()
for ncall_array in ncall_mean_list:
    plt.plot(burn_in_list, ncall_array,'-', label=r'MP')

plt.xlabel(r'Burn-in, $N_b$')
plt.ylabel(r'Number of calls to the LSF, $N_{call}$')
plt.tight_layout()

plt.show()
