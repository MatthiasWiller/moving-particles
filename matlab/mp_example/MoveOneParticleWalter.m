function [mean_pf,coeffOfvar_pf,numChains] = MoveOneParticleWalter(dim,betap,maxNLSFevals,ConditionalSamplingmode,nBurnIn,rhok)

% Original code by Iason applied to the
% dimension
%dim = 100;
disp(['Number of dimensions:', num2str(dim)]);
% number of samples at each level

% parameters
%rhok = 0.5;

% beta prior
%betap = 3.5;

% linear limit state
gfun = @(x) (-sum(x)/sqrt(dim) + betap); %(-sum(x)+betap*sqrt(dim));

% number of independent simulation runs
nRuns = 500;
for iRuns = 1: nRuns
    
    numsim = 0;
    ichain = 0;
    
    % set initial subset and failure level
    m = 0;
    while numsim < maxNLSFevals
        ichain = 1 + ichain;
        
        % Perform the first Monte Carlo simulation
        
        % Do the simulation (create array of random numbers)
        u = randn(dim,1);
        uk(:,1) = u;
        gk(1) = gfun(u);
        numsim = numsim+1;
        
        while 1
            
            % compute p0 percentile
            gp0 = gk;
            
            if gp0 <= 0
                Steps_m(iRuns,ichain) = m;
                break
            end
            
            m = m+1;
            
            if strcmp(ConditionalSamplingmode,'rejectionSampling') %Rejection Sampling
                numsim = numsim + 1; %Unfair since rejections are not counted as additional LSF evaluations
                while 1
                    u = randn(dim,1);
                    uk(:,1) = u;
                    gk(1) = gfun(u);
                    
                    if gk(1) < gp0
                        break
                    end
                end
            elseif strcmp(ConditionalSamplingmode,'MCMC') %MCMC algorithm proposed in Papaiannou et al 2015
                
               while 1
                    for iBurnin = 1:nBurnIn+1
                        u_cand = normrnd(rhok'.*uk(:,1)',sqrt(1-rhok.^2)');
                        g_u_cand = gfun(u_cand);
                        numsim = numsim+1;
                        if g_u_cand < gp0
                            gk = g_u_cand;
                            uk(:,1) = u_cand;
                        end
                    end %for
                    
                   if gk(1) < gp0 % According to walter et al 2016 generate a sample LARGER smaller than the previous (--> do not count repeated sample)
                        break
                   end
                    
                end
            end %if
            
            
        end
        
        clear gk gkf gp0 idx ind  numfpts p0 u ucand uf uk
        
        
        
    end
    
    NumberOfChains(iRuns) = ichain;
    pf = (1-1/(ichain))^m;
    pf_iRuns(iRuns) = pf;
    disp(['Run: ',num2str(iRuns),' #Chains: ', num2str(ichain)]);
end % for
% value
mean_pf = mean(pf_iRuns);
coeffOfvar_pf = std(pf_iRuns)/mean_pf;


%fprintf('\n***SubSim Pf: %g ***\n', pf);
Pf_exact = normcdf(-betap,0,1);
%fprintf('\n***Exact Pf: %g ***\n\n', Pf_exact);
numChains = mean(NumberOfChains);
end %function