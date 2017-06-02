"""
# ---------------------------------------------------------------------------
# Several utilities functions
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
def get_n_eff_sim(g_list):
    # initialization
    n_sim    = len(g_list)
    n_levels = np.zeros(n_sim, int)

    # count number of levels for every simulation
    for i in range(0, n_sim):
        n_levels[i] = len(g_list[i])

    # find most often encountered n_levels
    count_n_levels   = np.bincount(n_levels)
    most_often_level = np.nanargmax(count_n_levels)
    #n_levels         = most_often_level
    n_eff_sim = count_n_levels[most_often_level]

    return n_eff_sim