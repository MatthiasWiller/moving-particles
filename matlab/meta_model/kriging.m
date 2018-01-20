%% Init
% Data generating function
fh = @(x)(2*cos(2*pi*x/10).*x);
% range
x = -5:0.01:5;
N = length(x);
% Sampled data points from the generating function
M = 5;
selection = logical(zeros(N,1));
j = randsample(N, M);
% mark them
selection(j) = 1;
Xa = x(selection);

%% GP computations
% compute the function and extract mean
f = fh(Xa);
sigma2 = 2; % correlation length
sigma_noise = 0;
var_kernel = 10;
% computing the interpolation using all x's
% It is expected that for points used to build the GP cov. matrix, the
% uncertainty is reduced...
K = squareform(pdist(x'));
K = var_kernel*exp(-(0.5*K.^2)/sigma2);
% upper left corner of K
Kaa = K(selection,selection);
% lower right corner of K
Kbb = K(~selection,~selection);
% upper right corner of K
Kab = K(selection,~selection);
% mean of posterior
m = Kab'/(Kaa + sigma_noise*eye(M))*f';
% cov. matrix of posterior
D = Kbb - Kab'/(Kaa + sigma_noise*eye(M))*Kab;

%% Plot
figure;
grid on;
hold on;
% GP estimates
plot(x(~selection), m);
plot(x(~selection), m + 2*sqrt(diag(D)), 'g-');
plot(x(~selection), m - 2*sqrt(diag(D)), 'g-');
% Observations
plot(Xa, f, 'Ok');
% True function
plot(x, fh(x), 'k');