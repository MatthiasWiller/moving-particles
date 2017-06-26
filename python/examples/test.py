"""
Author: Matthias Willer 2017
"""

import matplotlib.pyplot as plt
import numpy as np

import plots.user_plot as uplt

theta = []
chain = []
point = np.array([1,1])
chain.append(point)
point = np.array([0,0])
chain.append(point)
point = np.array([3,5])
chain.append(point)


theta.append(chain)

# limit-state function
LSF = lambda u: np.minimum(3 + 0.1*(u[0] - u[1])**2 - 2**(-0.5) * np.absolute(u[0] + u[1]), 7* 2**(-0.5) - np.absolute(u[0] - u[1]))

uplt.plot_2d_contour_with_samples(theta, LSF)
plt.show()