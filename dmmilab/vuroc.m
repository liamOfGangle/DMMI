function v=vuroc(z,y)
% VUROC
% Computes Volume Under the ROC curve using the method of Hand & Till
% (Machine Learning, 45, 171-186, 2001) - AUROC in the binary case.
%
% Syntax: V = VUROC(Z,Y);
% Z - 1-of-C coded matrix of targets
% Y - matrix of probabilistic classifier outputs, 0<=y_ij<=1
%     vector Z & Y allowed for binary problems
%
% (c)2007 The University of Sheffield
%

if ~isequal(size(z),size(y)); error('matrix arguments must be same size');end
if any(max(y))>1 | any(min(y))<0;
    y=logistic(y); % place predictions on [0,1] monotonically
end
[r,c]=size(z);
if c==1;c=2;z=[z 1-z];y=[y 1-y];end %
a=zeros(c,c);v=0;
for ii=1:c
    for jj=1:c
        [tmp,ind]=sort(y(:,ii));
        tmp=z(ind,jj);
        n1=length(find(tmp==1));
        n0=length(find(tmp==0));
        a(ii,jj)=(sum(find(tmp==0))-n0*(n0+1)/2)/n0/n1;
    end
end
a=(a+a')/2;
v=2/(c-1)/c*sum(sum(triu(a,1)));

