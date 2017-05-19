"""
Author: Matthias Willer 2017
"""

import matplotlib.pyplot as plt
import scipy.stats as scps

#import plots.sus_plot as splt

# parameters
n_samples_per_level = 100
p0 = 0.1
g = []
p_F_SS = 0.00023

# analytical CDF
#beta = 3.5
#analytical_CDF = lambda x: scps.norm.cdf(x, beta)

#splt.plot_sus(g, p0, n_samples_per_level, p_F_SS, analytical_CDF)
#plt.show()
Ca = 135
print("0.000707852")
lam = 1.0
pf = 1 - scps.gamma.cdf(Ca, 100, 1/lam)
sf = scps.gamma.sf(Ca, 100)
print(pf)
print(sf)