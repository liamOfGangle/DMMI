%% 
% clear/close any variables/figures and load qsar.mat file
clear all
close all
load qsar.mat

z = D(:,11);
D(:,11) = [];
X = D; clear D;
%%
% choose model building parameters
n = floor(length(X)*(3/4)); % choose sample. 3:1 sample:oos split
k = 5; % number of k-fold

nhid = [4:2:16]; % array of # of HUs to try
nhid(1) = 5;nhid(end) = 15;

rho = logspace(0,1,50); % Broad logarray for rho between 10^0 and 10^1 with 50 equally spaced intervals

nits = 100; % number of iterations

outfunc = 'logistic';
%%
% randomly split data sets into sample and oos
[x,z,x_star,z_star] = dmrndsplit(X,z,n);
%%
% benchmark
myglm = glm(size(x,2),1,outfunc);
options=foptions;
options(1)=1;
options(14)=nits;
[myglm]=glmtrain(myglm,options,x,z); % train glm
y_star_hat=glmfwd(myglm,x_star); % evaluate oos
figure();dmroc(z_star,y_star_hat);
figure();dmscat(z_star,y_star_hat);
figure();dmplotres(z_star,y_star_hat)
figure();histfit(z_star-y_star_hat)
%%
% run through broad rho logspace with multiple # of HUs to determine 'best'
% # of HUs from the intial array
rho_HU = [];

for i=1:length(nhid)
    PI = zeros(length(rho),1);
    for j=1:length(rho)
       options = foptions;
       options(1) = 0;
       options(14) = nits;
       mymlp = mlp(size(x,2),nhid(i),1,outfunc,rho(j));
       [y_hat,Jemp] = dmxval(mymlp,options,x,z,k);
       PI(j) = Jemp;
    end

    idx = find(PI == min(PI));
    rho_min = rho(idx);

    Min = log(floor(rho_min))/log(10);
    Max = log(ceil(rho_min))/log(10);
    rho = logspace(Min,Max,100); % finer search with smaller intervals between Min and Max

    PI = zeros(length(rho),1);
    for j=1:length(rho)
        options = foptions;
        options(1) = 0;
        options(14) = nits;
        mymlp = mlp(size(x,2),nhid(i),1,outfunc,rho(j));
        [y_hat,Jemp] = dmxval(mymlp,options,x,z,k);
        PI(j) = Jemp;
    end
    
    temp = [rho_min,nhid(i)];
    rho_HU = vertcat(rho_HU,temp);
end
%%
% split rho_HU into rho_min and HU
rho_min = rho_HU(:,1);nhid = rho_HU(:,2);
%%
% retrain using rho_min
yValues = [];
PI = zeros(length(rho),1);
for i=1:length(rho_min)
    options=foptions; % initialize options
    options(1)=0; % set "silent"
    options(14)=nits; % ensure enough iterations allowed
    mymlp=mlp(size(x,2),nhid(i),1,outfunc,rho_min(i)); % initialize mymlp
    [mymlp,options]=mlptrain(mymlp,options,x,z); % train mlp
    y_hat=mlpfwd(mymlp,x); % evaluate on TRAINING sample
    y_star_hat=mlpfwd(mymlp,x_star); % evaluate oos
    mlp_arr(i) = mymlp;
    yValues = horzcat(yValues,y_star_hat);
    PI_arr(i) = options(8);
end
%%
% choose best model based off of minimum PI
idx = find(PI_arr == min(PI_arr));
y_star_hat = yValues(:,idx);
mymlp = mlp_arr(idx);
%%
% plot graphs
disp(['value of PI after re-training = ' num2str(PI_arr(idx))]);
figure();dmroc(z_star,y_star_hat)
figure();dmscat(z_star,y_star_hat);
figure();dmplotres(z_star,y_star_hat)
figure();histfit(z_star-y_star_hat)
%%
% save models, x_star and z_star
save dm150153458 myglm mymlp x_star z_star
