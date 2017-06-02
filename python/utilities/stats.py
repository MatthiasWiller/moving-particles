"""
# ---------------------------------------------------------------------------
# Several functions for statistical analysis
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-05
# ---------------------------------------------------------------------------
"""

import numpy as np

# ---------------------------------------------------------------------------
# compute cov analytically
def cov_analytical(theta, g, p0, N, pf_sus):
    m   = len(g)         # number of levels of the SubSim
    Nc  = int(p0 * N)    # number of Markov chains (= number of seeds)
    Ns  = int(1/p0)      # number of samples per Markov chain

    # initialization
    p       = np.zeros(m, float)
    b       = np.zeros(m, float)
    delta   = np.zeros(m, float)

    # compute intermediate failure levels
    for j in range(0, m):
        g_sort  = np.sort(g[j])
        b[j]    = np.percentile(g_sort, p0*100)
    #print("> > Last threshold =", b[m-1], "-> is now corrected to 0!")
    b[m-1] = 0    # set last threshold to 0

    # compute coefficient of variation for level 0 (MCS)
    delta[0] = np.sqrt(((1 - p0)/(N * p0)))             # [Ref. 1 Eq (3)]

    # compute coefficient of variation for other levels
    for j in range(1, m):

        # compute indicator function matrix for the failure samples
        I_Fj = np.reshape(g[j] <= b[j], (Ns, Nc))

        # sample conditional probability (~= p0)
        p_j = (1/N)*np.sum(I_Fj[:, :])
        print("> > p_j [", j, "] =", p_j)

        # correlation factor (Ref. 2 Eq. 10)
        gamma = np.zeros(Ns, float)

        # correlation at lag 0
        sums = 0
        for k in range(0, Nc):
            for ip in range(1, Ns):
                sums += (I_Fj[ip, k] * I_Fj[ip, k])   # sums inside [Ref. 1 Eq. (22)]
        R_0 = (1/N)*sums - p_j**2    # autocovariance at lag 0 [Ref. 1 Eq. (22)]
        print("R_0 =", R_0)

        # correlation factor calculation
        R = np.zeros(Ns, float)
        for i in range(0, Ns-1):
            sums = 0
            for k in range(0, Nc):
                for ip in range(0, Ns - (i+1)):
                    sums += (I_Fj[ip, k] * I_Fj[ip + i, k])         # sums inside [Ref. 1 Eq. (22)]
            R[i]     = (1/(N - (i+1)*Nc)) * sums - p_j**2               # autocovariance at lag i [Ref. 1 Eq. (22)]
            gamma[i] = (1 - ((i+1)/Ns)) * (R[i]/R_0)                    # correlation factor [Ref. 1 Eq. (20)]

        gamma_j = 2*np.sum(gamma)                                 # [Ref. 1 Eq. (20)]
        print("gamma_j =", gamma_j)

        delta[j] = np.sqrt(((1 - p_j)/(N * p_j)) * (1 + gamma_j)) # [Ref. 2 Eq. (9)]

    # compute resulting cov
    delta_sus = np.sqrt(np.sum(delta**2))

    return delta_sus
