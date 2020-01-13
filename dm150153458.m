%%
%
% Written and tested on MATLAB 2019b
% Some functions provided by Professor Rob Harrison
%
%
%% 
% clear/close any variables/figures and load qsar.mat file
clear all
close all
load qsar.mat

% split data into output z and input matrix X
z = D(:,end);
D(:,end) = [];
X = D; clear D;
%%
% choose model building parameters
n = floor(length(X)*(3/4)); % choose sample. 3:1 sample:oos split
k = 5; % number of k-fold cross validations

% create an array of number of hidden units - [5 6 8 10 12 14 15]
nhid = (4:2:16); % array of # of HUs to try 
nhid(1) = 5;nhid(end) = 15;

rho = logspace(0,1,50); % Broad logarray for rho between 10^0 and 10^1 with 50 equally spaced intervals

nits = 100; % number of iterations

outfunc = 'logistic';
%%
% randomly split data sets into sample and oos
[x,z,x_star,z_star] = dmrndsplit(X,z,n);
%%
% create a generalised linear model to use as a benchmark against a mutilayer perceptron 
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
rho_HU = []; % intialise an empty array for rho_min and # of HUs for that rho_min

for i=1:length(nhid) % run through nhid array
    % perform cross validated training on 'broad' rho array
    PI = zeros(length(rho),1);
    for j=1:length(rho) % run through first rho array
       options = foptions; % initialise options
       options(1) = 0; % set silent
       options(14) = nits;
       mymlp = mlp(size(x,2),nhid(i),1,outfunc,rho(j)); % intialise mlp for nhid(i) and rho(j)
       [y_hat,Jemp] = dmxval(mymlp,options,x,z,k);
       PI(j) = Jemp; % use Jemp as performance indicator
    end
    
    idx = find(PI == min(PI)); % find the index location of the minimum performance indicator
    rho_min = rho(idx); % find the rho at that index location
    
    % create a logspace using the two closest integers to the previous rho
    Min = log(floor(rho_min))/log(10); % lower integer
    Max = log(ceil(rho_min))/log(10); % higher integer
    rho = logspace(Min,Max,100); % finer search with smaller intervals between Min and Max
    
    % repeat of for loop above but with a finer logspace of rho 
    PI = zeros(length(rho),1);
    for j=1:length(rho)
        options = foptions;
        options(1) = 0;
        options(14) = nits;
        mymlp = mlp(size(x,2),nhid(i),1,outfunc,rho(j));
        [y_hat,Jemp] = dmxval(mymlp,options,x,z,k);
        PI(j) = Jemp;
    end
    
    idx = find(PI == min(PI)); % find the index location of the minimum performance indicator
    rho_min = rho(idx); % find the rho at that index location
    PI_pretrain(i) = PI(idx); % place PI pretraining into an array to compare with post training PIs 
    % place the rho_min for that # of HUs into an array 
    temp = [rho_min,nhid(i)];
    rho_HU = vertcat(rho_HU,temp);
end
%%
% split rho_HU into rho_min and HU
rho_min = rho_HU(:,1);nhid = rho_HU(:,2);
%%
% retrain all mlps of different # of HUs using rho_min found in previous section
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
    mlp_arr(i) = mymlp; % place model into struct array
    yValues = horzcat(yValues,y_star_hat); % place y_star_hat values into array
    PI_arr(i) = options(8); % place PI value into array 
end
%%
% choose best model based off of minimum PI
idx = find(PI_arr == min(PI_arr));
y_star_hat = yValues(:,idx); % pick y_star_hat values from mlp with best PI
mymlp = mlp_arr(idx); % pick mlp model with best PI
%%
% plot graphs
disp(['value of PI after re-training = ' num2str(PI_arr(idx))]);
figure();dmroc(z_star,y_star_hat) % AUROC graph
figure();dmscat(z_star,y_star_hat); 
figure();dmplotres(z_star,y_star_hat)
figure();histfit(z_star-y_star_hat)
%%
% save models, x_star and z_star
save dm150153458 myglm mymlp x_star z_star
