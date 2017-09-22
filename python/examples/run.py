"""
Author: Matthias Willer 2017
"""

import time as timer

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scps
import scipy.integrate as integrate
import scipy.interpolate as interpolate


np.random.seed(0)

# f1 = lambda y, x: scps.multivariate_normal.pdf(np.array([x,y]))
def f1(y,x):
    return scps.multivariate_normal.pdf(np.array([x,y]),mean=np.array([0,0]))
n_steps = 100.0

x_line       = np.linspace(-4, 4, n_steps)
f_marg   = np.zeros(len(x_line))

for i in range(0, len(x_line)):
    f_marg[i], err = integrate.quad(f1, -20, 20, args=(x_line[i]))

# normalize
f_marg = f_marg/np.sum(f_marg)
F_marg = np.cumsum(f_marg)

num_CDF = lambda x: interpolate.spline(x_line,F_marg,x)
inv_CDF = lambda x: interpolate.spline(F_marg,x_line,x)


# F_analytical = scps.norm.cdf(x_new)
print('true:', scps.norm.cdf(x_new))
print('num:', num_CDF(x_new))



plt.figure()
plt.plot(x_line, F_marg)
# plt.plot(x, F_analytical)

plt.show()