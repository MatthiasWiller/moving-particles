clc; clear; close all;

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
x_new = x(~selection);
f = fh(Xa);

%% GP computations
% compute the function and extract mean


[m, D] = gpr(Xa,f,x_new);

%% Plot
figure;
grid on;
hold on;
% GP estimates
plot(x_new, m);
plot(x_new, m + 2*sqrt(diag(D)), 'g-');
plot(x_new, m - 2*sqrt(diag(D)), 'g-');
% Observations
plot(Xa, f, 'Ok');
% True function
plot(x, fh(x), 'k');