#--------------------------------------------------------------------------
# Adaptive Conditional Sampling
# Created on Mon Feb 27 14:49:34 2017
# @author: felipe.uribe@tum.de
#--------------------------------------------------------------------------
# Input:
# * G       : limit state function in the standard space
# * u_tilde : seeds used to generate the new samples
# * b       : actual intermediate level
# * Ns      : number of seeds
# * Nc      : number of chains per seed
# * lam     : scaling parameter lambda 
#--------------------------------------------------------------------------
# Output:
# * u_jp1 : next level samples
# * h_jp1 : limit state function evaluations of the new samples
# * lam   : next scaling parameter lambda
#--------------------------------------------------------------------------
# Based on:
# 1."MCMC algorithms for subset simulation"
#    Papaioannou et al.
#    Probabilistic Engineering Mechanics 41 (2015) 83-103.
#--------------------------------------------------------------------------

#  Initial import
import scipy as sp
import numpy as np

def CS(G, u_tilde, b_j, Ns, Nc, lam):
    
    # create empty list to store output
    u_jp1 = list()
    h_jp1 = list()
    a_jp1 = list()
    
    # some constants
    pa     = 0.1
    a_star = 0.44
    d  = np.size(u_tilde, axis=1)   # dimension
    Na = pa*Ns                      # number of chains for adaptation
    a_hat = np.zeros([Na,1])    
    
    # 1. Estimate initial standard deviation from the seeds     
    mu_hat    = np.zeros([d,1])
    sigma_hat = np.zeros([d,1])
    for k in range(d):
        for i in range(Ns):
            mu_hat[k]    = mu_hat[k] + u_tilde[i,k]
            sigma_hat[k] = sigma_hat[k] + (u_tilde[i,k]-mu[k])**2
        mu_hat[k]    = mu_hat[k]/Ns
        sigma_hat[k] = np.sqrt(mu_hat[k]/(Ns-1))
    
    sigma = min(1,lam*sigma_hat)
    rho   = np.sqrt(1-sigma**2)
    
    # 2. Randomize the ordering of the seeds to avoid vias
    idx = np.random.permutation(Ns)
    u_tilde = u_tilde[idx]
    
    # 3. ACS procedure
    a_bar = list()   # store acc/rej values until adaptation
    for i in range(Ns):
        # initialize storage for chains of the i-th seed
        uu = np.zeros(Nc,d)
        hh = np.zeros(Nc,1)
        aa = np.ones(Nc,1)
        
        # first values   
        uu[0,:] = u_tilde[i,:]
        hh[0]   = G(uu)
        aa[0]   = 1

        for p in range(Nc-1):
            # sample from the conditional density
            u_star = np.multiply(rho,uu) + np.multiply(sigma,np.random.randn(d,1))
            # evaluate LSF
            s = G(u_star)
            if (s <= b_j):
                uu[p+1,:] = u_star
                hh[p+1]   = s
                aa[p+1]   = 1
            else:
                uu[p+1,:] = uu[p]
                hh[p+1]   = hh[p]
                aa[p+1]   = 0 
                
        u_jp1.append(uu)
        h_jp1.append(hh)
        a_jp1.append(aa)
        a_bar.append(aa)
        
        if mod(i,Na)==0:
            t = int(i/Na)
            a_hat[t] = np.mean(a_bar, axis=0)
            lam[t]   = np.exp(np.log(lam[t-1]) + (a_hat[t]-a_star)/np.sqrt(t) )
            sigma    = min(1,lam[t]*sigma_hat)
            rho      = np.sqrt(1-sigma**2)
            # empty the list a_bar
            a_bar = list()
        
    return u_jp1, h_jp1, lam[:]