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

savepdf = True
example = 3


# ---------------------------------------------------------------------------
# DEFINE FUNCTIONS
# ---------------------------------------------------------------------------

# Monte Carlo
monte_carlo = lambda pf, delta: (1-pf)/(pf*delta**2)

# Subset Simulation
p0 = 0.1
subset = lambda pf, delta, gamma: ((np.log(pf)/np.log(p0)*(1-p0)/np.sqrt(p0))**2 + np.log(pf)/np.log(p0)*(1-p0)/p0)*((1+gamma)/(delta**2))

# Moving particles
Nb = 5
moving_particles = lambda pf, delta, Nb: (Nb*(-np.log(pf))+1)*(-np.log(pf))/(np.log(delta**2 + 1))



if example == 1:
    # analytical CDF
    d      = 10     
    beta   = 3.0902       # for pf = 10^-3
    analytical_CDF = lambda x: scps.norm.cdf(x, beta)
    pf_ref = analytical_CDF(0)

    example_name = 'example_1_d10'

elif example == 2:
    pf_ref = 2.275e-3
    example_name = 'waarts'

elif example == 3:
    pf_ref = 3.17e-5
    example_name = 'breitung'

elif example == 4:
    pf_ref = 4.42e-3
    example_name = 'liebscher'

# ---------------------------------------------------------------------------
# LOAD RESULTS FROM SIMULATIONS
# ---------------------------------------------------------------------------
# -- load sus data ----------------------------------------------------------
direction = 'python/data/example' + repr(example) + '/nsamples_study_sus/'
nsamples_list_sus = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,\
                     2000, 3000, 4000, 5000]

ncall_points_sus = np.zeros(len(nsamples_list_sus))
cov_points_sus = np.zeros(len(nsamples_list_sus))

for i in range(0, len(nsamples_list_sus)):
    N = nsamples_list_sus[i]
    g_list_list_sus = \
        np.load(direction + 'sus_' + example_name + '_N' + repr(N) + '_Nsim100_cs_g_list.npy')

    ncall_points_sus[i] = \
        uutil.get_mean_ncall_from_SUS(g_list_list_sus, N, p0)

    pf_temp, cov_points_sus[i] = \
        uutil.get_mean_and_cov_pf_from_SUS(g_list_list_sus, N, p0)


# -- load mp data ----------------------------------------------------------
direction = 'python/data/example' + repr(example) + '/nsamples_study_mp/'
nsamples_list_mp = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, \
                30, 40, 50, 60, 70, 80, 90, 100]

ncall_points_mp = np.zeros(len(nsamples_list_mp))
cov_points_mp = np.zeros(len(nsamples_list_mp))

for i in range(0, len(nsamples_list_mp)):
    N = nsamples_list_mp[i]
    g_list_list_mp = \
        np.load(direction + 'mp_' + example_name + '_N' + repr(N) + '_Nsim100_b5_cs_sss2_g_list.npy')

    ncall_points_mp[i] = \
        uutil.get_mean_ncall_from_MP(g_list_list_mp, N, Nb)

    pf_temp, cov_points_mp[i] = \
        uutil.get_mean_and_cov_pf_from_MP(g_list_list_mp, N)

# -- load mp data ----------------------------------------------------------
max_cov = max([max(cov_points_mp), max(cov_points_sus)])
delta_line = np.linspace(0.05, max_cov, 100)

ncall_line_mcs = monte_carlo(pf_ref, delta_line)
ncall_line_sus = subset(pf_ref, delta_line, 1.5)
ncall_line_mp  = moving_particles(pf_ref, delta_line, Nb)
# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------


plt.figure()
plt.plot(ncall_line_mcs, delta_line, color='C0', label='MCS')
plt.plot(ncall_line_sus, delta_line, color='C1', label='SuS')
plt.plot(ncall_line_mp, delta_line, color='C2', label='MP')

plt.plot(ncall_points_sus, cov_points_sus, '.', color='C1')
plt.plot(ncall_points_mp, cov_points_mp, '.', color='C2' )

# legend
plt.legend()

# xaxis
plt.xscale('log')
plt.xlabel(r'Computational cost, $N_{call}$')

# yaxis
plt.ylabel(r'Coefficient of variation, $\delta$')

plt.tight_layout()
if savepdf:
    plt.savefig('cov_over_ncall.pdf', format='pdf', dpi=50, bbox_inches='tight')

plt.show()