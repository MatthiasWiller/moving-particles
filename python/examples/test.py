"""
Author: Matthias Willer 2017
"""

import matplotlib.pyplot as plt
import scipy.stats as scps
import numpy as np

import plots.user_plot as uplt

#import plots.sus_plot as splt


#sf = scps.gamma.sf(Ca, 100)
#f = scps.beta.pdf(-0.0043, 6, 6)
#print(f)

#j = scps.expon.pdf(0.26, scale=1/1.0)
#print(j)

z = lambda x:  (8* np.exp(- (x[0]**2 + x[1]**2)) + 2* np.exp(-((x[0]-5)**2 + (x[1]-4)**2)) + 1 + x[0]*x[1]/10)

x       = np.linspace(-2, 6, 100)
X, Y    = np.meshgrid(x, x)
Z       = z([X, Y])
# plt.figure()
# plt.contour(X, Y, Z)

plt.figure()
ax = plt.gca(projection='3d')
ax.plot_surface(X, Y, Z)

#uplt.plot_surface_custom(z)
plt.show()