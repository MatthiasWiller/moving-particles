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
n_samples_per_level = 500      # SUS: number of samples per conditional level
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

# analytical CDF
# analytical_CDF = lambda x: scps.norm.cdf(x, beta)

# pf_analytical    = analytical_CDF(0)

color_list = ['blue', 'green', 'orange', 'brown', 'red']

# ---------------------------------------------------------------------------
# LOAD RESULTS FROM SIMULATIONS
# ---------------------------------------------------------------------------
direction = 'python/data/'

theta_list_sus = np.load(direction + 'sus_example_1_d2_N500_Nsim20_acs_theta_list.npy')
g_list_sus     = np.load(direction + 'sus_example_1_d2_N500_Nsim20_acs_g_list.npy')


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


    theta = np.array(theta_list_sus[sim])
    g     = np.array(g_list_sus[sim])
    n_lvl = theta.shape[0]

    
    
    for i in range(0, n_lvl):
        fig = plt.figure()
        plt.axes().set_aspect('equal')

        plt.contour(X, Y, Z, [0], colors='k')
        for lvl in range(0, i+1):
            if lvl < n_lvl-1: # don't plot contour for last level
                g_lvl     = g[lvl,:]
                b_lvl = np.percentile(g_lvl, p0*100)
                plt.contour(X,Y,Z,[b_lvl], colors='k')

            theta_lvl = theta[lvl,:,:]
            plt.scatter(theta_lvl[:, 0], theta_lvl[:, 1], s=1, color=color_list[lvl], marker='o', linestyle='None')

        # set labels
        plt.xlabel(r'$u_1$')
        plt.ylabel(r'$u_2$')

        plt.xlim(-6, 6)
        plt.ylim(-6, 6)

        plt.xticks([-5, 0, 5])
        plt.yticks([-5, 0, 5])

        plt.tight_layout()
        
        if savepdf:
            plt.savefig('example_1_sim' + repr(sim) +'_sus_lsf_w_samples_2D_lvl' + repr(i+1) + '.pdf', format='pdf', dpi=50, bbox_inches='tight')

    plt.show()

