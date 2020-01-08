%%
% clear/close any variables/figures and load in qsar.mat file.
clear all 
close all
load qsar.mat 

z = D(:,11);
D(:,11) = [];
%%
% Choose model building paramaters

n = floor(length(D)*(2/3)); % choose sample size
nhid = [1 3 5 10 15]; % number of hidden units
k = 5; % choose k for k-fold cv

PI_arr = zeros(length(nhid),1);
rho_arr = zeros(length(nhid),1);

rho = logspace(0,1,100); % grid rho. Intial search is broad
nits = 200; % number of iterations

outfunc = 'logistic';
%%
[x,z,x_star,z_star] = dmrndsplit(X,z,n);
%%
for l=1:length(nhid)
    
    PI = zeros(length(rho),1);
    for i=1:length(rho)
        options = foptions;
        options(1) = 0;
        options(14) = nits;
        mymlp = mlp(size(x,2),nhid(l),1,outfunc,rho(i));
        [y_hat,Jemp] = dmxval(mymlp,options,x,z,k);
        PI(i) = Jemp;
    end
    idx = find(PI == min(PI));
    rho_min = rho(idx);
    
    Min = log(floor(rho_min))/log(10);
    Max = log(ceil(rho_min))/log(10);
    rho = logspace(Min,Max,200); % finer search for rho based off of the lowest value
    PI = zeros(length(rho),1);
    for j=1:length(rho)
        options = foptions;
        options(1) = 0;
        options(14) = nits;
        mymlp = mlp(size(x,2),nhid(l),1,outfunc,rho(j));
        [y_hat,Jemp] = dmxval(mymlp,options,x,z,k);
        PI(j) = Jemp;
    end
    idx = find(PI == min(PI));
    PI_arr(l) = PI(idx);rho_arr(l) = rho(idx);
end

% figure(1);semilogx(rho,PI);xlabel('ln (\rho)');ylabel('Jemp');
% idx = find(PI == min(PI));
% rho_min = rho(idx); rho_old = rho_min;
% PI_old = PI(idx);
% disp(['best value for rho = ' num2str(rho_min)])
% disp(['minimum value of PI = ' num2str(PI(idx))])
%%
% retrain using rho_min 
retrain_PI_arr = zeros(length(nhid),1);
for l=1:length(rho_arr)
    options = foptions;
    options(1) = 0;
    options(14) = nits;
    mymlp=mlp(size(x,2),nhid(l),1,outfunc,rho_arr(l)); % initialize mymlp
    [mymlp,options]=mlptrain(mymlp,options,x,z); % train mlp
    y_hat=mlpfwd(mymlp,x); % evaluate on TRAINING sample
    y_star_hat=mlpfwd(mymlp,x_star); % evaluate oos
    
    figure(l);dmroc(z_star,y_star_hat);
    retrain_PI_arr(l) = options(8);
end
    

%%
% figure(2);semilogx(rho,PI);xlabel('ln (\rho)');ylabel('Jemp');
% idx = find(PI == min(PI));
% rho_min = rho(idx); rho_new = rho_min;
% PI_new = PI(idx);
% disp(['best value for rho = ' num2str(rho_min)])
% disp(['minimum value of PI = ' num2str(PI(idx))])
%%
% % retrain using rho_min;
% options=foptions; % initialize options
% options(1)=0; % set "silent"
% options(14)=nits; % ensure enough iterations allowed
% mymlp=mlp(size(x,2),nhid,1,outfunc,rho_min); % initialize mymlp
% [mymlp,options]=mlptrain(mymlp,options,x,z); % train mlp
% y_hat=mlpfwd(mymlp,x); % evaluate on TRAINING sample
% y_star_hat=mlpfwd(mymlp,x_star); % evaluate oos
% %%
% disp(['value of PI after re-training = ' num2str(options(8))])
% figure(3);dmroc(z_star,y_star_hat)

