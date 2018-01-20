function y_pce = pce_full(X,Y,p)
%{
--------------------------------------------------------------------------
Max Ehre
September 2017
--------------------------------------------------------------------------
compute coefficients of full d-dimensional PCE of order p based on
OLS of (X,Y), where X ~ Gauss(0,1).
--------------------------------------------------------------------------
modules

- multi_index.m
- regres.m
--------------------------------------------------------------------------
%}
tic

[n_s,d_y] = size(Y);
[~, d_x] = size(X);

% symbolic Hermite polynomials
syms xx;

He_s(1) = sym(1);
He_s(2) = xx;
for j = 2:p+1
   He_s(j+1) = expand(xx*He_s(j) - (j-1)*He_s(j-1));
end

% create symbolic vector for transformed space x and original space y
x = sym('x',[1 d_x],'real');

% multi-index
alpha  = multi_index(d_x,p);
P = length(alpha);

% multi-dimensional Hermite polynomials
psi = arrayfun(@(i) prod(diag(subs(He_s(alpha(i,:)+1),x'))),1:P,'uniform',0);
psi = [psi{:}]'./sqrt(prod(factorial(alpha),2));

% information matrix
args = num2cell(X,1);
func = matlabFunction(psi(2:end)','Vars',x);
Psi = [ones(n_s,1) func(args{:})];

% regression
[R_sq,Q_final,kappa,a,pce] = regres(Psi,X,Y,psi);   

% export PCE
f = matlabFunction(sym(pce),'Vars',{x'},'file','f_pce');

% evaluate PCE at ED
y_pce = mm_pce_full(X')';

t_pce = toc

end