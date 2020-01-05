function [y,Jemp]=dmxval(net,options,x,t,m)
% MFOLDCV
% Conducts basic m-fold cross validation on Netlab structures
%
% Syntax: [Y,Jemp] = XVAL(NET,OPTIONS,X,T,M)
% NET, OPTIONS, X, T follow Netlab conventions
% M    - number of "folds" or sub-samples
% Y    - matrix of outputs
% Jemp - cross-validated, empirical cost appropriate to task
%
% Limitations:
%   MLPs use 'scg' optimizer by default - edit MLPTRAIN to vary this
%   sub-samples NOT stratified
%
% (c)2012 The University of Sheffield
%

if nargin<5;error('five inputs required');end
if m==1;error('one is an invalid no. of folds');end
toptions=options; % retain copy of original options vector

% shuffle data to break any artificial ordering
[NS,NC]=size(t);
rand('state',512); % fix random number seed for repeatibility of shuffle
ind=randperm(NS);

S=fix(NS/m); % length of subsamples
ntype=net.type; % get type of network

y=NaN*ones(size(t));
if strcmp(ntype,'kda');y(:,end)=[];end
for ii=1:m
    % choose indices of testing set
    if ii<m
        exc=(ii-1)*S+1 : ii*S;
    else
        exc=(ii-1)*S+1 : NS; % may have a different number of samples if number of samples not integer multiple of m
    end
    inc=setdiff(1:NS,exc); % choose indices of training set
    eval(['[tnet,options]=' ntype 'train(net,toptions,x(ind(inc),:),t(ind(inc),:));']); % train network
    eval(['[yt]=' ntype 'fwd(tnet,x(ind(exc),:));']); % compute network output
    y(ind(exc),:)=yt; % save network outputs
end

switch net.outfn

  case 'linear'  	% Linear outputs
    Jemp = 0.5*sum(sum((y - t).^2));

  case 'logistic'  	% Logistic outputs
    Jemp = - sum(sum(t.*log(y) + (1 - t).*log(1 - y)));

  case 'softmax'   	% Softmax outputs
    Jemp = - sum(sum(t.*log(y)));

  otherwise
    error(['Unknown activation function ', net.outfn]);
end
