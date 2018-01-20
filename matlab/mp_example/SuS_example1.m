function [mean_pf,coeffOfvar_pf,avrgnLSFevals] = SuS_example1(dim,betap)
% Original code by Iason applied to the
% dimension
%dim = 100;
disp(['Number of dimensions:', num2str(dim)]);
% number of samples at each level
nsamlev = 1000;

% maximum number of levels
maxlev = 20;

% parameters
p0 = 0.1;
rhok = 0.8;

% beta prior
%betap = 3.5;

numsim = 0;

% number of independent simulation runs
nRuns = 500;
for iRuns = 1: nRuns; disp(['Run: ',num2str(iRuns)]);
    
    % initialize samples
    uk=zeros(dim,nsamlev);
    
    % linear limit state
    gfun = @(x) (-sum(x)/sqrt(dim) + betap); %(-sum(x)+betap*sqrt(dim));
    

    
    % Perform the first Monte Carlo simulation
    for k=1:nsamlev
        
        % Do the simulation (create array of random numbers)
        u = randn(dim,1);
        
        uk(:,k) = u;
        
        gk(k) = gfun(u);
        
        numsim = numsim+1;
        
        
    end
    
    
    % set initial subset and failure level level
    m = 1;
    gp0 = 1.0;
    
    
    
    
    while 1
        
        
        % compute p0 percentile
        gp0      = prctile(gk,p0*100);
 %       fprintf('\n-Threshold level = %g \n', gp0);
        [gk,ind] = sort(gk);
        uk       = uk(:,ind);
        
        if gp0 <= 0 || m == maxlev
            
            % compute the failure probability
            if gp0 > 0
                
%                disp('Maximum number of levels reached');
                disp('The probability computed is an approximate upper bound');
%                
                pf = p0^m;
                
            else
                
                numfpts = sum(gk <= 0);
                
                pf = p0^(m-1)*numfpts/nsamlev;
                
            end
            
            break
            
        end
        
        % find failure points at current level
        idx = gk <= gp0;
        gkf = gk(idx);
        uf  = uk(:,idx);
        

        % number of failure points
        numfpts = length(gkf);
        
        % permute to get rid of depenence
        ind = randperm(numfpts);
        gkf = gkf(ind);
        uf  = uf(:,ind);
        
        % compute length of each chain
        lenchain = floor(nsamlev/numfpts);
        
        lenchaink = lenchain*ones(numfpts,1);
        lenchaink(1:mod(nsamlev,numfpts))=lenchain+1;
        
        
        % initialize counters
        count = 0;
        counta = 0;
        

        % delete previous samples
        gk = [];
        uk = [];
        
        for k=1:numfpts
            count = count+1;
            
            % set seed for chain and accept as first sample
            uk(:,count) = uf(:,k);
            gk(count) = gkf(k);
            
            for j = 2:lenchaink(k)
                count = count+1;
                
                % get candidate sample from conditional normal distribution
                ucand = normrnd(rhok'.*uk(:,count-1)',sqrt(1-rhok.^2)');
                
                % Evaluate limit-state function
                gk(count) = gfun(ucand); 
                numsim    = numsim+1;
                
                % check if sample is accepted
                if gk(count) <= gp0
                    uk(:,count) = ucand;
                else
                    uk(:,count) = uk(:,count-1);
                    gk(count) = gk(count-1);
                end
            end
            
            
        end
        

        
        m = m+1;
        
    end
    
pf_irun(iRuns) = pf;

end %for

% value
mean_pf = mean(pf_irun);
coeffOfvar_pf = std(pf_irun)/mean_pf;

% average number of LSF evaluations
avrgnLSFevals = numsim/nRuns;


%fprintf('\n***SubSim Pf: %g ***\n', pf);
Pf_exact = normcdf(-betap,0,1);
%fprintf('\n***Exact Pf: %g ***\n\n', Pf_exact);

end %function