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


# -------------------------------------------------------------

# The interpolation step:


# % define g values
# gg = 0:0.01:20;   % predefined depending on the problem
# figure; hold on;
# for i = 1:size(g_curve,2)
#    [xN, index] = unique(g_curve(:,i));
#    yN          = pf_curve(index,i);
#    pp(:,i)     = interp1(xN,yN,gg);
#    semilogy(gg,pp(:,i));
# end
# pf95  = prctile(pp',97.5);
# pf05  = prctile(pp',2.5);
# pfmu  = mean(pp,2);
# pfstd = std(pp,1,2);



# -------------------------------------------------------------

# The SDOF LSF function:

# SDOF linear oscillator with:
# omega = 7.85;            % Natural frequency [rad/s]
# zeta  = 0.02;            % Damping ratio [adim]

# % And system equation:
# A   = [0,1; -omega^2,-2*zeta*omega];
# B   = [0,1]';
# C   = [1,0];
# D   = 0;
# x0  = [0,0]';
# sys = ss(A,B,C,D);       % System in state space formulation 

# % Subject to a white noise excitation with:
# S  = 1;                  % White noise spectral intensity 
# T  = 30;                 % Duration of the excitation, s
# dt = 0.02;               % Time increment, s
# t  = (0:dt:T)';          % discrete time instants
# n  = length(t);          % n points ~ number of random variables
# % The uncertain state vector theta consists of the sequence of i.i.d.
# % standard Gaussian random variables which generate the white noise input
# % at the discrete time instants

# Initial data
# % Configuration parameters
# N  = 1000;        % Total number of samples for each level
# p0 = 0.1;         % Probability of each subset, chosen adaptively

# % Max demand, marginal PDFs of the random variables and limit-state function
# B  = 2.1;   % critical threshold
# mu = 0;   sigma = 1;
# pi_rnd = @() normrnd(mu,sigma,n,1);
# g      = @(theta) max(abs(lsim(sys, sqrt(2*pi*S/dt)*theta, t, x0, 'zoh')'));
# #END
