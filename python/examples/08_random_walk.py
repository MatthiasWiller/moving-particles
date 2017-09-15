"""
# ---------------------------------------------------------------------------
# Script to simulate a random walk
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
matplotlib.rcParams.update({'font.size': 23})
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams['text.usetex'] = True

from matplotlib import cm
from matplotlib import ticker   

import scipy.stats as scps


# parameters

beta = -3.5
n_runs = 100
burnin = 40
sigma_cond = 0.8

# initialization
m_list = []
chain_list = []


# sampling
for i in range(0, n_runs):
    u_list = []
    u = np.random.randn() # draw sample
    u_list.append(u)
    u_old = u
    m = 0
    while u_old > beta:
        b = burnin
        u_tmp = u_old
        while b > 0:
            print('b =',b)
            u = np.random.normal(u_tmp, sigma_cond, 1) # draw conditional sample
            if u < u_old:
                u_tmp = u
            b = b - 1
        u_list.append(u_tmp)
        u_old = u_tmp
        m = m + 1
        print('m =', m)

        # u = np.random.normal(u_old, sigma_cond, 1) # draw conditional sample
        # if u < u_old:
        #     u_list.append(u)
        #     u_old = u
        #     m = m + 1

    m_list.append(m)
    chain_list.append(u_list)

m_sum = sum(m_list)

pf_analytical = scps.norm.cdf(0, -beta)
pf_estimator = (1 - 1/n_runs)**m_sum

print("pf_analytical =", pf_analytical)
print("pf_estimator =", pf_estimator)

# plot
#fig = plt.figure()
