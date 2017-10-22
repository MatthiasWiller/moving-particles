function [R,Q_cor,kappa,coeff,model] = regres(Psi,X,Y,psi)
%{
--------------------------------------------------------------------------
Max Ehre 
June 2017
--------------------------------------------------------------------------
perform regression of target values Y w/ predictors psi, 
experimental design X and Matrix Psi
--------------------------------------------------------------------------
Input:

Psi:  Psi_ij = psi_j(X_i)
.
.
.
--------------------------------------------------------------------------
Definitions:

d is dimensionality of input X
P is size pf psi, i.e. # of regressors
N is # of training points
--------------------------------------------------------------------------
%}

    [N,d] = size(X);
    P = size(psi,1);

    % create symbolic vector
    x = sym('x',[1 d],'real');
    
    M = Psi'*Psi;
    coeff = M\(Psi'*Y);
    kappa = cond(M);

    model = matlabFunction(coeff'*psi,'Vars',x);

    args  = num2cell(X,1);
    hatY  = model(args{:})';
    
    hatY  = hatY';
    InvM  = inv(M);
    R     = 1 - var(Y-hatY,1)/var(Y);
    h     = diag(Psi*InvM*Psi');
    Q     = 1 - mean(((Y - hatY)./(1-double(h))).^2)/var(Y);
    
    % corrected LOO measure-of-fit for small EDs acc. to Chappelle et al. (2002)
    Q_cor = 1 - (1+trace(InvM))/(1-P/N)*mean(((Y - hatY)./(1-double(h))).^2)/var(Y);
end

