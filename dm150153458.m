%%
% clear/close any variables/figures and load in qsar.mat file.
clear all 
close all
load qsar.mat 

z = D(:,11);
D(:,11) = [];
X = D;
%%
% Choose model building paramaters

n = floor(length(D)*(3/4)); % choose sample size
nhid = 10; % number of hidden units
k = 5; % choose k for k-fold cv

rho = logspace(-2,2,100); % grid rho. Intial search is broad
nits = 100; % number of iterations

outfunc = 'logistic';
%%
[x,z,x_star,z_star] = dmrndsplit(X,z,n);
%%
PI = zeros(length(rho),1);
for i=1:length(rho)
    options = foptions;
    options(1) = 0;
    options(14) = nits;
    mymlp = mlp(size(x,2),nhid,1,outfunc,rho(i));
    [y_hat,Jemp] = dmxval(mymlp,options,x,z,k);
    PI(i) = Jemp;
end

figure(1);semilogx(rho,PI);xlabel('ln (\rho)');ylabel('Jemp');
idx = find(PI == min(PI));
rho_min = rho(idx);
disp(['best value for rho = ' num2str(rho_min)])
disp(['minimum value of PI = ' num2str(PI(idx))])
