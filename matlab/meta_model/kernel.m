function x_new = kernel(x)
% conditional sampling
sigma_cond = 0.6;
rho_k = 0.8;
d = length(x);
W = randn(d, 1);
x_new = zeros(d, 1);
for k = 1:d
  mu_cond = rho_k * x(k);
  x_new(k) = normrnd(mu_cond, sigma_cond);
end
% x_new = (x + sigma_cond.*W)./sqrt(1 + sigma_cond*sigma_cond);
