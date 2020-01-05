function [X,m,s]=dmstandard(X,m,s)
% [X,m,s]=dmstandard(X,m,s)
%
% Performs standardization on data matrix
% data must be 'simple', i.e. no column-wise relationship
%
% INPUTS:
% X - input data
% m - mean from an earlier operation to be used on a new smaple
% s - std dev from an earlier operation to be used on a new smaple
%
% OUPUTS:
% X - standardized input data
% m - mean of original data (for future use on oos or operational data)
% s - std dev of original data (for future use on oos or operational data)
%
% USAGE:
% [X,m,s]=dmstandard(X) - standardizes X & returns the standardized matrix
%                       & mean & std dev for future use
%
% [X]=dmstandard(X,m,s) - standardizes X using previously computed mean & std dev
%                       & returns the standardized matrix
%
%
if nargin~=3 & nargin~= 1; error('must be either 1 or 3 input arguments'); end
if nargin==1
m=mean(X);
s=std(X);
end
X=(X-repmat(m,size(X,1),1))./repmat(s,size(X,1),1);
