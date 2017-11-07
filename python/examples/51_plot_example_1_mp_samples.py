"""
# ---------------------------------------------------------------------------
# File to produce plots for example 5
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

import time as timer

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 20
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

np.random.seed(0)

# ---------------------------------------------------------------------------
# STANDARD INPUT
# ---------------------------------------------------------------------------

savepdf = True

# parameters
n_samples_per_level = 1000      # SUS: number of samples per conditional level
p0                  = 0.1       # SUS: Probability of each subset, chosen adaptively

n_initial_samples   = 100       # MP: Number of initial samples 

n_sim = 1

# limit-state function
d = 2
#beta = 5.1993       # for pf = 10^-7
#beta = 4.7534       # for pf = 10^-6
#beta = 4.2649       # for pf = 10^-5
#beta = 3.7190       # for pf = 10^-4
beta = 3.0902       # for pf = 10^-3
#beta = 2.3263       # for pf = 10^-2
LSF  = lambda u: u.sum(axis=0)/np.sqrt(d) + beta

level_list = [0, 4, 9]
color_list = ['blue','green', 'orange']

# ---------------------------------------------------------------------------
# LOAD RESULTS FROM SIMULATIONS
# ---------------------------------------------------------------------------
direction = 'python/data/'

#g_list_mp      = np.load(direction + 'mp_breitung_N100_Nsim20_b20_cs_sss2_g_list.npy')
theta_list_mp  = np.load(direction + 'mp_example_1_d2_N100_Nsim5_b20_cs_sss2_theta_list.npy')


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
for sim in range(0, n_sim):
    print('sim =', sim+1)

    # get grid, minimum, maximum
    x       = np.linspace(-6, 6, 300)
    X, Y    = np.meshgrid(x, x)
    Z       = LSF(np.array([X, Y]))

    min_x = min(X.flatten())
    min_y = min(Y.flatten())
    max_x = max(X.flatten())
    max_y = max(Y.flatten())

    # get m_max
    theta_list = theta_list_mp[sim]
    m_vec = (len(list_tmp) for list_tmp in theta_list)
    m_max = max(m_vec)

    fig = plt.figure()
    plt.axes().set_aspect('equal')
    plt.contour(X, Y, Z, [0], colors='k')

    for i in range(0, len(level_list)):
        lvl = level_list[i]
        theta = []
        for theta_tmp in theta_list:
            if len(theta_tmp) > lvl:
                theta.append(theta_tmp[lvl])

        theta = np.array(theta)
        plt.scatter(theta[:, 0], theta[:, 1], s=1, color=color_list[i], marker='o', linestyle='None')

    # plot all failure samples
    theta = []
    for theta_tmp in theta_list:
        theta.append(theta_tmp[-1])

    theta = np.array(theta)
    plt.scatter(theta[:, 0], theta[:, 1], s=1, color='red', marker='o', linestyle='None')
    # set labels
    plt.xlabel(r'$u_1$')
    plt.ylabel(r'$u_2$')

    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    plt.xticks([-5, 0, 5])
    plt.yticks([-5, 0, 5])
    
    plt.tight_layout()
    if savepdf:
        plt.savefig('example_1_sim' + repr(sim) +'_lsf_w_samples_2D.pdf', format='pdf', dpi=50, bbox_inches='tight')


plt.show()
