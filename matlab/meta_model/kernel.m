function x_new = kernel(x)
% conditional sampling
sigma_cond = 0.6;
W = randn(length(x), 1);
x_new = (x + sigma_cond.*W)./sqrt(1 + sigma_cond*sigma_cond);
