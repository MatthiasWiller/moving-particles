"""
# ---------------------------------------------------------------------------
# COV plots
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-06
# ---------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scps

import utilities.plots as uplt

# ---------------------------------------------------------------------------
# COV plots for example 1
# ---------------------------------------------------------------------------

# 50 effective simulations, 1000 samples per level, 10 dimensions
# aCS: pa = 0.1, CS: rho_k = 0.8

# beta
beta        = np.array([2.3263    , 3.0902    , 3.7190    , 4.2649    , 4.7534    , 5.1993    ])
# pf        =          [10^-2     , 10^-3     , 10^-4     , 10^-5     , 10^-6     , 10^-7     ]
pf = scps.norm.cdf(-beta)

cov_mmh_est = np.array([0.18608372, 0.23967790, 0.31903349, 0.35812461, 0.48023234, 0.56572077])
cov_cs_est  = np.array([0.17821342, 0.25565418, 0.33789344, 0.37837155, 0.55967960, 0.61266539])
cov_acs_est = np.array([0.17030047, 0.23853249, 0.34479622, 0.35877576, 0.41551040, 0.41706365])

cov_mmh_ana = np.array([0.17505588, 0.23675370, 0.34361219, 0.36072132, 0.41243377, 0.48704470])
cov_cs_ana  = np.array([0.19644757, 0.24787005, 0.31314800, 0.33635469, 0.38632857, 0.41213623])
cov_acs_ana = np.array([0.19523141, 0.23491152, 0.28483936, 0.33690432, 0.37247420, 0.41486202])

# plot 
# estimated
uplt.plot_cov_over_pf(pf, cov_mmh_est, cov_cs_est, cov_acs_est)
# analytical
uplt.plot_cov_over_pf(pf, cov_mmh_ana, cov_cs_ana, cov_acs_ana)
plt.show()
