function [X,Z,X_star,Z_star]=dmrndsplit(inp,tar,n)
% [X,Z,X_star,Z_star]=dmrndsplit(inp,tar,n)
%
% makes a random split of a set of input/target data
% INP - matrix of input data (columns are variables)
% TAR - matrix of target data (columns are outputs)
% N   - number of samples required in training sample
%
% 
if nargin <2; error('Insufficient input arguments');end
n_tot=size(inp,1);nn=size(tar,1);
if n_tot ~= nn;error('Input & Target matrices must have same # of rows');end
if nargin<3;n=floor(n_tot/2);disp('approx. 50/50 split assumed');end
if n_tot <= n; error('Requested training sample larger than total # samples');end

ind=randperm(size(inp,1));
X=inp(ind(1:n),:);
Z=tar(ind(1:n),:);
X_star=inp(ind(n+1:end),:);
Z_star=tar(ind(n+1:end),:);
