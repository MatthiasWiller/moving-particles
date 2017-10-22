function [pf, Ncall] = mp(LSF, b, N, Nb)
d=2;
uu = zeros(N,d);
g = zeros(N,1);
Ncall = 0;
% MCS
for i = 1:N
  uu(i,:) = normrnd(0, 1.0, [1, d]);
  g(i) = LSF(uu(i,:)); Ncall = Ncall + 1;
end

m = 0;
while max(g) > b
  [gmax, imax] = max(g);
  gold = gmax;
  % select seed randomly
  irand = randi(N);
  uold = uu(irand, :);
  
  for i = 1:Nb
    ustar = kernel(uold')';
    gstar = LSF(ustar); Ncall = Ncall + 1;
    if gstar < gmax
      uold = ustar;
      gold = gstar;
    end
  end
  uu(imax,:) = uold;
  g(imax)    = gold;
  m = m+1;
end

pf = (1-1/N)^m;
