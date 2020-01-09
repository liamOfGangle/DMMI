%% 
% clear variables/figures and load in data set
clear all;
close all;
load qsar.mat;
%%
% manipulate data set
z = D(:,11);  % place column 11 into array z
D(:,11) = []; % remove col 11 from D
X = D;        % rename D to X
%%
% choose model building parameters
n = floor(length(X)*0.75);   % choose sample size. 3:1 sample:oos split
nhid = [1 5 10 15 20 25 30 35 40 45 50]; % # of HU to try

Broad = 10;
Fine = 50;

rho = logspace(0,1,Broad); % grid rho. Broad search between 1 and 10

k = 5;      % k for k-fold cv
nits = 100; % number of iterations

outfunc = 'logistic';
%%
% randomly split data into sample:oos
[x,z,x_star,z_star] = dmrndsplit(X,z,n);
%%
% run through all # HUs with a broad and then fine (but still a large interval gap) rho grid search. 
for h=1:length(nhid)
    PI = zeros(length(rho),1);
    for i=1:length(rho)
        options = foptions;
        options(1) = 0;
        options(14) = nits;
        mymlp = mlp(size(x,2),nhid(h),1,outfunc,rho(i));
        [y_hat,Jemp] = dmxval(mymlp,options,x,z,k);
        PI(i) = Jemp;
    end
    
    idx = find(PI == min(PI));
    rho_min = rho(idx);
    
    % closest integer values based off the broad minimum search
    Min = log(floor(rho_min))/log(10);
    Max = log(ceil(rho_min))/log(10);
    rho = logspace(Min,Max,Fine); % finer search between Min and Max integers
    
    PI = zeros(length(rho),1);
    for i=1:length(rho)
        options = foptions;
        options(1) = 0;
        options(14) = nits;
        mymlp = mlp(size(x,2),nhid(h),1,outfunc,rho(i));
        [y_hat,Jemp] = dmxval(mymlp,options,x,z,k);
        PI(i) = Jemp;
    end
    
    idx = find(PI == min(PI));
    
    C = [PI(idx),rho(idx),nhid(h)];
    min_check = vertcat(min_check,C);
    
    rho_min = rho(idx);
    
    % retrain using rho_min;
    options=foptions; % initialize options
    options(1)=0; % set "silent"
    options(14)=nits; % ensure enough iterations allowed
    mymlp=mlp(size(x,2),nhid(h),1,outfunc,rho_min); % initialize mymlp
    [mymlp,options]=mlptrain(mymlp,options,x,z); % train mlp
    y_hat=mlpfwd(mymlp,x); % evaluate on TRAINING sample
    y_star_hat=mlpfwd(mymlp,x_star); % evaluate oos
    
    disp(['value of PI after re-training = ' num2str(options(8))])
    figure(h);dmroc(z_star,y_star_hat)
    
    D = [options(8),nhid(h)];
    min_retrain = vertcat(min_retrain,D);
end








    
    
    
    
    
    
