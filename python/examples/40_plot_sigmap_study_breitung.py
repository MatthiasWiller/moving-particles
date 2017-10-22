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

import utilities.plots as uplt
import utilities.util as uutil

matplotlib.rcParams.update({'font.size': 20})
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams['text.usetex'] = True

print("RUN file")

# set seed for randomization
np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT
# ---------------------------------------------------------------------------

savepdf = True
example = 3

sampling_method = 'cs'
algorithm = 'mp' # sus
                  # mp
                  # susvsmp

# parameters
n_samples_per_level = 1000      # SUS: number of samples per conditional level
p0                  = 0.1       # SUS: Probability of each subset, chosen adaptively

n_initial_samples   = 100       # MP: Number of initial samples


if sampling_method == 'cs':
    sigma_p_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sigma_p_list_sus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    mtxt = r'CS'

elif sampling_method == 'mmh':
    sigma_p_list = [0.1, 0.2, 0.3, 0.4, 0.5, \
                    0.6, 0.7, 0.8, 0.9, 1.0, \
                    1.2, 1.4, 1.6, 1.8, 2.0, \
                    2.2, 2.4, 2.6, 2.8, 3.0, \
                    3.2, 3.4, 3.6, 3.8, 4.0, \
                    4.2, 4.4, 4.6, 4.8, 5.0]
    mtxt = r'MMH'

n_sigma = len(sigma_p_list)


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
direction_sus = 'python/data/example' + repr(example) + '/sigma_p_study_sus/'
direction_mp = 'python/data/example' + repr(example) + '/sigma_p_study_mp/'

g_list_list_sus = []
g_list_list_mp = []

iii = 0
for sigma_p in sigma_p_list_sus:
    iii = iii + 1
    g_list_mp_tmp = np.load(direction_sus + 'sus_' + example_name + \
                     '_N' + repr(n_samples_per_level) + '_Nsim100_' + sampling_method + \
                     '_sigmap' + repr(iii) + '_g_list.npy')
    g_list_list_sus.append(g_list_mp_tmp)

iii = 0
for sigma_p in sigma_p_list:
    iii = iii + 1
    g_list_mp_tmp = np.load(direction_mp + 'mp_' + example_name + \
                    '_N' + repr(n_initial_samples) + '_Nsim100_b5_' + sampling_method + \
                    '_sss2_sigmap' + repr(iii) + '_g_list.npy')
    g_list_list_mp.append(g_list_mp_tmp)


# ---------------------------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------------------------

# initialization
pf_line_list  = []
b_line_list   = []
cov_line_list = []
legend_list   = []
cov_at_pf_array_sus = np.zeros(n_sigma-1, float)
pf_mean_array_sus = np.zeros(n_sigma-1, float)

cov_at_pf_array_mp = np.zeros(n_sigma, float)
pf_mean_array_mp = np.zeros(n_sigma, float)


#  subset simulation
for i in range(0, n_sigma-1):
    pf_mean_array_sus[i], cov_at_pf_array_sus[i] = \
        uutil.get_mean_and_cov_pf_from_SUS(g_list_list_sus[i], n_samples_per_level, p0)

# moving particles
for i in range(0, n_sigma):
    pf_mean_array_mp[i], cov_at_pf_array_mp[i] = \
        uutil.get_mean_and_cov_pf_from_MP(g_list_list_mp[i], n_initial_samples)

# analytical expression
pf_ref  = np.ones(n_sigma, float) * pf_ref
cov_ref = np.ones(n_sigma, float) * np.sqrt(pf_ref**(-1/n_initial_samples)-1)


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

# plot cov over standard deviation
plt.figure()

if algorithm == 'mp':
    plt.plot(sigma_p_list, cov_ref,'-', label=r'MP analytical', color='C0')
    plt.plot(sigma_p_list, cov_at_pf_array_mp,'+-', label=r'MP with '+mtxt, color='C1')
elif algorithm == 'sus':
    plt.plot(sigma_p_list_sus, cov_at_pf_array_sus,'x-', label=r'SuS with '+mtxt, color='C2')
elif algorithm == 'susvsmp':
    plt.plot(sigma_p_list, cov_ref,'-', label=r'MP analytical', color='C0')
    plt.plot(sigma_p_list, cov_at_pf_array_mp,'+-', label=r'MP with '+mtxt, color='C1')
    plt.plot(sigma_p_list_sus, cov_at_pf_array_sus,'x-', label=r'SuS with '+mtxt, color='C2')


plt.legend()
plt.xlabel(r'Standard deviation, $\sigma_P$')
plt.ylabel(r'Coefficient of variation, $\hat{\delta}_{p_f}$')
if example == 3:
    plt.xlim([.05, 1.05])


plt.tight_layout()
if savepdf:
    plt.savefig('sigma_p_study_cov_over_sigma_' + sampling_method + '_' + algorithm + '.pdf', format='pdf', dpi=50, bbox_inches='tight')

# plot pf over d
plt.figure()
if algorithm == 'mp':
    # plt.plot(sigma_p_list, pf_ref,'-', label=r'Analytical', color='C0')
    plt.plot(sigma_p_list, pf_ref,'-', label=r'Reference (MCS)', color='C0')
    plt.plot(sigma_p_list, pf_mean_array_mp,'+-', label=r'MP with '+mtxt, color='C1')
elif algorithm == 'sus':
    # plt.plot(sigma_p_list, pf_ref,'-', label=r'Analytical', color='C0')
    plt.plot(sigma_p_list, pf_ref,'-', label=r'Reference (MCS)', color='C0')
    plt.plot(sigma_p_list_sus, pf_mean_array_sus,'x-', label=r'SuS with '+mtxt, color='C2')
elif algorithm == 'susvsmp':
    # plt.plot(sigma_p_list, pf_ref,'-', label=r'Analytical', color='C0')
    plt.plot(sigma_p_list, pf_ref,'-', label=r'Reference (MCS)', color='C0')
    plt.plot(sigma_p_list, pf_mean_array_mp,'+-', label=r'MP with '+mtxt, color='C1')
    plt.plot(sigma_p_list_sus, pf_mean_array_sus,'x-', label=r'SuS with '+mtxt, color='C2')

plt.yscale('log')
if example == 1:
    plt.ylim([6e-4, 2e-3])
if example == 2:
    plt.ylim([2e-3, 3e-3])
if example == 3:
    plt.xlim([.05, 1.05])
    plt.ylim([1e-5, 1e-4])
if example == 4:
    plt.ylim([3e-3, 6e-3])

plt.legend()
plt.xlabel(r'Standard deviation, $\sigma_P$')
plt.ylabel(r'Probability of failure, $\hat{p}_f$')

plt.tight_layout()
if savepdf:
    plt.savefig('sigma_p_study_pf_over_sigma_' + sampling_method + '_' + algorithm + '.pdf', format='pdf', dpi=50, bbox_inches='tight')

plt.show()
