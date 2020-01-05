function y=dmcat2bin(z)
% CAT2BIN
% Converts a vector of n categorical values to one-from-n-1 binary matrix
% i.e. dummy encoding removing redundancy
%

[N,tmp]=size(z);
if tmp>1; error('Input must be categorical vector');end
v=unique(z);n=length(v);
y=zeros(N,n);
for ii=1:n
    ind=find(z==v(ii));
    y(ind,ii)=1;
end
y(:,end)=[];