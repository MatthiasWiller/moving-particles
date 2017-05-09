"""
Author: Matthias Willer 2017
"""

import numpy as np

import time as timer

# theta0  : inital state of a Markov chain 
# N       : total number of states, i.e. samples
# target_PDF
# proposal_PDF
# LSF     : limit state function g(x)

def modified_metropolis(theta0, N, marginal_PDF, sample_prop_PDF, f_prop_PDF, LSF, b):
  #startTime = timer.time()  
  
  # get dimension
  d = np.size(theta0)

  # initialize theta
  theta       = np.zeros((N, d), float)
  theta[0, :] = theta0

  xi = np.zeros((d), float)

  for i in range(1, N):
    # generate a candidate state xi:
    for k in range(0, d):
      # sample xi from proposal_PDF
      xi[k] = sample_prop_PDF(theta[i-1, k])

      # compute acceptance ratio

      # alpha = (p(y) * q(y,x)) /   =   (p(y) * g(y)) /
      #         (p(x) * q(x,y))         (p(x) * g(x))
      alpha = (marginal_PDF(xi[k])          * f_prop_PDF(theta[i-1, k], xi[k]))/ \
              (marginal_PDF(theta[i-1, k])  * f_prop_PDF(xi[k], theta[i-1, k]))

      r     = np.minimum(alpha, 1)

      # accept or reject xi by setting ...
      if np.random.uniform(0, 1) <= r:
        # accept
        xi[k] = xi[k]
      else:
        # reject
        xi[k] = theta[i, k]
    
    # check whether xi is in Failure domain (system analysis) and accept or reject xi
    if LSF(xi) <= b:
      # in failure domain -> accept
      theta[i, :] = xi
    else:
      # not in failure domain -> reject
      theta[i, :] = theta[i-1, :]
  
  # output
  #print("> > > Time needed for MMH =", round(timer.time() - startTime, 2), "s")
  return theta
