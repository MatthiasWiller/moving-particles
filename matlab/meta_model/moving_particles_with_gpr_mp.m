clc;
clear;
close all;
rng(0);

%% input parameters
Nsim = 10;
Nfail = 10;
Nb = 20;
q = -4;

Nmp = 1000;

% limit state function
d = 2;

% example 1
% LSF = @(u) sum(u)/sqrt(d) + 3.0902;

% example 2
LSF = @(u) min(3 + 0.1*(u(1) - u(2))^2 - 2^(-0.5) * abs(u(1) + u(2)), 7* 2^(-0.5) - abs(u(1) - u(2)));

% example 3
% LSF = @(u) min(5-u(1), 1/(1+exp(-2*(u(2)+4)))-0.5);

% initialization
Ncall_vec = zeros(1, Nsim);
pf_vec = zeros(1, Nsim);

%% Algorithm 8
for sim = 1:Nsim
  disp(['Starting simulation ', num2str(sim)]);
  X = normrnd(0,1,[d,1]);
  y = LSF(X); Ncall = 1;
  tbl = table(X(1), X(2), y);
  % sample d+1 points and calculate g
  for i = 1:9
    X = normrnd(0,1,[d,1]);
    y = LSF(X); Ncall = Ncall + 1;
    row = table(X(1), X(2), y);
    tbl = [tbl;row];
  end
  % learn meta-model 
  gprMdl = fitrgp(tbl,'y');
  for i = 1:Nfail
    % sample X and evaluate LSF
    Xm = normrnd(0,1,[d,1]); 
    ym = LSF(Xm); Ncall = Ncall + 1;
    y = ym;
    % add Xm to table
    row = table(Xm(1), Xm(2), y);
    tbl = [tbl;row];
    % train meta-model
    gprMdl = fitrgp(tbl,'y');
    while y > q
      Xmm = Xm; ymm = ym;
      for j = 1:Nb
        Xstar = kernel(Xmm);
        % evaluate meta-model
        ystar = predict(gprMdl, Xstar');
        % accept if lower LSF-value
        if ystar < ymm
          Xmm = Xstar; 
          ymm = ystar;
        end
      end
      % evaluate LSF
      ymm = LSF(Xmm); Ncall = Ncall + 1;
      y = ymm;
      % train meta-model
      row = table(Xmm(1), Xmm(2), y);
      tbl = [tbl;row];
      gprMdl = fitrgp(tbl,'y');
      % accept if lower LSF-value
      if ymm < ym
        Xm = Xmm;
        ym = ymm;
      end
    end
    disp('converged!')
  end
  clear tbl;
  fprintf('end model learning\n')
  
  Ncall_vec(sim) = Ncall;
  
  LSF_model = @(u) predict(gprMdl, u);
  
  % mp of model
  [pf, n] = mp(LSF_model, q, Nmp, 20);
  pf_vec(sim) = pf;
end

pf_mean = mean(pf_vec);
pf_std = std(pf_vec);
pf_cov = pf_std/pf_mean;

disp(['pf_mean =', num2str(pf_mean)]);
disp(['pf_cov =', num2str(pf_cov)]);
