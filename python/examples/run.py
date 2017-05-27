"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import scipy.stats as scps

np.random.seed(0)
lam = 1.0
f_marg_PDF_1     = lambda x: scps.expon.pdf(x, scale=1/lam)
f_marg_PDF_2     = lambda x: lam * np.exp(-lam*x)

x = -1.5
print(f_marg_PDF_1(x))
print(f_marg_PDF_2(x))