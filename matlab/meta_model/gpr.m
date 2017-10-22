function [m, D] = gpr(x_data, f_data, x_new)
% get size of input data
[d, M] = size(x_data);
x = [x_data x_new];

sigma2 = 2; % correlation length
sigma_noise = 0;
var_kernel = 10;

% computing the interpolation using all x's
% It is expected that for points used to build the GP cov. matrix, the
% uncertainty is reduced...
K = squareform(pdist(x'));
K = var_kernel*exp(-(0.5*K.^2)/sigma2);
% upper left corner of K
Kaa = K(1:M,1:M);
% lower right corner of K
Kbb = K(M+1:end,M+1:end);
% upper right corner of K
Kab = K(1:M,M+1:end);
% mean of posterior
m = Kab'/(Kaa + sigma_noise*eye(M))*f_data';
% cov. matrix of posterior
D = Kbb - Kab'/(Kaa + sigma_noise*eye(M))*Kab;
