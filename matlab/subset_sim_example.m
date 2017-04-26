% Subset Simulation for Linear Reliability Problem
% Performance function: g(x)=x1+...+xd
% Input variables x1,...,xd are i.i.d. N(0,1)
% Written by K.M. Zuev, Institute of Risk & Uncertainty, Uni of Liverpool

clear;
d=1000;                     % dimension of the input space
YF=200;                     % critical threshold (failure <=>g(x)>YF)
pF=1-normcdf(YF/sqrt(d));   % true value of the failure probability
n=100;                      % number of samples per level
p=0.1;                      % level probability
nc=n*p;                     % number of Markov chains
ns=(1-p)/p;                 % number of states in each chain
L=0;                        % current (unconditional) level
x=randn(d,n);               % Monte Carlo samples
nF=0;                       % number of failure samples

for i=1:n
    y(i)=sum(x(:,i));       % system response y=g(x)
    if y(i)>YF              % y(i) =>YF> x(:,i) is a failure sample
        nF=nF+1;
    end
end

while nF(L+1)/n < p         % stopping criterion
    L=L+1;                  % next conditional lelvel is needed
    [y(L,:),ind]=sort(y(L,:),'descend');% renumbered responses
    x(:,:,L)=x(:,ind(:),L); % renumbered samples
    Y(L)=(y(L,nc)+y(L,nc+1))/2; % Lˆth intermediate threshold
    z(:,:,1)=x(:,1:nc,L);   % Markov chain "seeds"


    % Modified Metropolis algorithm for sampling from pi(x|F L)
    for j=1:nc
        for m=1:ns
            % Step 1:
            for k=1:d
                a=z(k,j,m)+randn; % Step 1(a)
                r=min(1,normpdf(a)/normpdf(z(k,j,m))); % Step 1(b)
                
                % Step 1(c):
                if rand < r
                    q(k)=a;
                else
                    q(k)=z(k,j,m);
                end
            end

            % Step 2:
            if sum(q)>Y(L)          % q belongs to F L
                z(:,j,m+1)=q;
            else
                z(:,j,m+1)=z(:,j,m);
            end
    
        end
    end
    
    for j=1:nc
        for m=1:ns+1
            x(:,(j-1)*(ns+1)+m,L+1)=z(:,j,m); % samples from pi(x|F_L)
        end
    end
    clear z;

    nF(L+1)=0;
    for i=1:n
        y(L+1,i)=sum(x(:,i,L+1));   % system response y=g(x)
        if y(L+1,i)>YF              % then x(:,i,L+1) is a failure sample
            nF(L+1)=nF(L+1)+1;      % number of failure samples at level L+1
        end
    end
end

pF_SS=p^(L)*nF(L+1)/n;              % SS estimate
N=n+n*(1-p)*(L);                    % total number of samples