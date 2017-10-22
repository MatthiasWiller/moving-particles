clc; clear all; close all;

%% input parameters
Nsim = 10;
q = -4;

Nmp = 2000;
%% LSF
% example 1
% LSF = @(u) sum(u)/sqrt(d) + 3.0902;

% example 2
LSF = @(u) min(3 + 0.1*(u(1) - u(2))^2 - 2^(-0.5) * abs(u(1) + u(2)), 7* 2^(-0.5) - abs(u(1) - u(2)));

% example 3
% LSF = @(u) min(5-u(1), 1/(1+exp(-2*(u(2)+4)))-0.5);

% initialization
Ncall_vec = zeros(1, Nsim);
pf_vec = zeros(1, Nsim);


for i = 1:Nsim
  [pf, Ncall] = mp(LSF, q, Nmp, 10);
  Ncall_vec(i) = Ncall;
  pf_vec(i) = pf;
end

pf_mean = mean(pf_vec);
pf_std = std(pf_vec);
pf_cov = pf_std/pf_mean;

disp(['pf_mean =', num2str(pf_mean)]);
disp(['pf_cov =', num2str(pf_cov)]);