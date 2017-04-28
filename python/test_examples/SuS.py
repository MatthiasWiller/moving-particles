#--------------------------------------------------------------------------  
# Subset simulation in the U-Space
# Created on Mon Feb 27 14:49:34 2017
# @author: felipe.uribe@tum.de
#--------------------------------------------------------------------------
# Input:
# * G : limit state function in the U-space
# * N : number of samples per conditional level
# * d : number of dimensions (i.e. number of basic variables)
# * p0: conditional failure probability value
#--------------------------------------------------------------------------
# Output:
# * Pf_SuS : failure probability estimate
#--------------------------------------------------------------------------
# Based on:
# 1."MCMC algorithms for subset simulation"
#    Papaioannou et al.
#    Probabilistic Engineering Mechanics 41 (2015) 83-103.
#--------------------------------------------------------------------------

#  Initial import
import scipy as sp
import numpy as np

def SuS(G, N, d, p0):

    # initializations and constants
    mit = 20
    Nf  = np.zeros((mit,1))       # store number of failure samples per level
    b   = np.zeros((mit,1))       # store number of intermediate levels  
    Ns  = int(N*p0)      # number of seeds per level
    Nc  = int(1/p0)      # number of samples simulated from each seed
    
    # initial MCS step
    j     = 0      # initial level
    Nf[j] = 0      # initial number of failure samples
    u0    = np.random.randn(N,d)    # N idd realizations from the std Gaussian
    h0    = np.zeros((N,1))
    for i in range(N):
        h0[i] = G(u0[i,:])       # print "sample %d --- G=%g \n" % (i,h0[i])
        if (h0[i] <= 0):
           Nf[j] = Nf[j]+1
    
    
    # SuS step
    while Nf[j] < Ns:
        # sort the limit state function values in ascending order
        h_prime = np.sort(h0)
        idx = sorted(range(len(h0)),key=lambda x:h0[x])
          
        # order the samples according to the previous order        
        u_prime = u0[(idx)]
  
        # compute the intermediate threshold level
        b[j] = 0.5*(h_prime[Ns-1]+h_prime[Ns])
        print("\nIntermediate threshold level b=%g \n" % b[j])
    
        # select the seeds for the MCMC sampler
        u_tilde = u_prime[0:Ns]
        
        # use an MCMC sampler to draw the conditional samples
        # I AM HERE
        u,h,lam = CS(G, u_tilde, b[j], Ns, Nc, lam)
        
        # check the number of failure samples 
        Nf[j+1]
        j += 1        
    
    Pf_SuS = (p0**j) * (Nf[j]/N)
    
    return Pf_SuS, u