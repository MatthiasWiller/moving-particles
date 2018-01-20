clc;
clear;

p = 1e-10;
N = linspace(10,1000, 200);

cov = sqrt(p.^(-1./N) - 1);

plot(N, cov)