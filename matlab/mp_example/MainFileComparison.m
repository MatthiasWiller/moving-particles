% Comparison
% 1: original SUS as implemented by Iason
% 2: Move one Particle Walter
% 
% 
Dims = 20;%[5,20,50];

betap = 3.5;
rhos = [0,0.2,0.4,0.6,0.8];
nBurnIns = 0:10;
betastr = num2str(betap); betastr(betastr == '.') = '_';

pf_exact = normcdf(-betap);

avrgnLSFevals = 3700;


for iDim = 1:numel(Dims)
    dim = Dims(iDim);
        [mu_pf.(['Beta',betastr]).('SuS').(['Dim',num2str(dim)]),cov_pf.(['Beta',betastr]).('SuS').(['Dim',num2str(dim)]),avrgnLSFevals] = SuS_example1(dim,betap);
        mu_pfSuS = mu_pf.(['Beta',betastr]).('SuS').(['Dim',num2str(dim)]);
        cov_pfSuS = cov_pf.(['Beta',betastr]).('SuS').(['Dim',num2str(dim)]);
       [mu_pf.(['Beta',betastr]).('MoveOneParticleWalter').('UncorrelatedSamples').(['Dim',num2str(dim)]),cov_pf.(['Beta',betastr]).('MoveOneParticleWalter').('UncorrelatedSamples').(['Dim',num2str(dim)])] = MoveOneParticleWalter(dim,betap,avrgnLSFevals,'rejectionSampling',[]);
    for iRho = 1:numel(rhos)
        rho_MCMC = rhos(iRho);disp(['RhoMCMC: ',num2str(rho_MCMC)])
        rho_MCMCstr{iRho} = num2str(rho_MCMC); rho_MCMCstr{iRho}(rho_MCMCstr{iRho} == '.') = '_';
        for iBurnIn = 1:numel(nBurnIns);%1:10;
            nBurnIn = nBurnIns(iBurnIn);disp(['BurnIn: ',num2str(nBurnIn)])
            [mu_pf.(['Beta',betastr]).('MoveOneParticleWalter').(['BurnIn',num2str(nBurnIn),'rho',rho_MCMCstr{iRho}]).(['Dim',num2str(dim)]),cov_pf.(['Beta',betastr]).('MoveOneParticleWalter').(['BurnIn',num2str(nBurnIn)]).(['Dim',num2str(dim)]),NumberOfChains(iRho,iBurnIn)] = MoveOneParticleWalter(dim,betap,avrgnLSFevals,'MCMC',nBurnIn,rho_MCMC);
            ExpectedError(iRho,iBurnIn) = mu_pf.(['Beta',betastr]).('MoveOneParticleWalter').(['BurnIn',num2str(nBurnIn),'rho',rho_MCMCstr{iRho}]).(['Dim',num2str(dim)])/pf_exact;
            CoefficientOfVariation(iRho,iBurnIn) = cov_pf.(['Beta',betastr]).('MoveOneParticleWalter').(['BurnIn',num2str(nBurnIn)]).(['Dim',num2str(dim)]);
            
        end
      save('/home/era3/Documents/Kilian/Error_vsRho_vsBUrnIn')  
    end %for
    
end%for
 
 
figure('Name','Expected Error for different SampleCorrelation and BurnIn (compared to SUS with same number of LSF evals), rho = 3.5')
hold on
ylabel('$E\left( \frac{pf}{\hat{pf}}\right)$','Interpreter','LaTex','FontSize',14)
xlabel('Burn In MCMC step','FontSize',14)
plot(nBurnIns,repmat(mu_pfSuS,size(nBurnIns)))
for iPlot = 1:size(ExpectedError,1)
    plot(nBurnIns,ExpectedError(iPlot,:))
end
legend('SuS 3700 LSF evals', ['Rho:',rho_MCMCstr{1}],['Rho:',rho_MCMCstr{2}],['Rho:',rho_MCMCstr{3}],['Rho:',rho_MCMCstr{4}],['Rho:',rho_MCMCstr{5}],'FontSize',14)

figure('Name','Coefficient of variation, rho = 3.5')
hold on
ylabel('c.o.v','FontSize',14)
xlabel('Burn In MCMC step','FontSize',14)
plot(nBurnIns,repmat(cov_pfSuS,size(nBurnIns)))
for iPlot = 1:size(CoefficientOfVariation,1)
    plot(nBurnIns,CoefficientOfVariation(iPlot,:))
end
legend('SuS 3700 LSF evals', ['Rho: ',num2str(rhos(1))],['Rho: ',num2str(rhos(2))],['Rho: ',num2str(rhos(3))],['Rho: ',num2str(rhos(4))],['Rho: ',num2str(rhos(5))],'FontSize',14)


figure('Name','Number of Chains, rho = 3.5')
hold on
ylabel('Number of Chains possible with 3700 LSF evaluations','FontSize',14)
xlabel('Burn In MCMC step','FontSize',14)
for iPlot = 1:size(CoefficientOfVariation,1)
    plot(nBurnIns,NumberOfChains(iPlot,:))
end
legend(['Rho: ',num2str(rhos(1))],['Rho: ',num2str(rhos(2))],['Rho: ',num2str(rhos(3))],['Rho: ',num2str(rhos(4))],['Rho: ',num2str(rhos(5))],'FontSize',14)

