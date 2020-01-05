function dmplot2D(net,X,z,k)
% plot2ddb(net,X,z,thresh)
% Plots a decision boundary in 2-D generated by an matlab MLP object.
%
% net    - a trained netlab object with 2 inputs and one output
% X      - 2-D input data
% z      - corresponding target data
% k      - the threshold value (posterior probability value at which decision is made)
%
% (c) 20 12The University of Sheffield

[r,c]=size(X); if c~=2;error('only works for 2D input data');end
minmax=[min([X])' max([X])'];npts=500;
[X1,X2]=meshgrid(linspace(minmax(1,1),minmax(1,2),npts),linspace(minmax(2,1),minmax(2,2),npts));
x=[X1(:) X2(:)];
if net.type=='mlp'
    y=mlpfwd(net,x);
elseif net.type=='glm'
    y=glmfwd(net,x);
elseif net.type=='rbf'
    y=rbffwd(net,x);
elseif net.type=='klr'
    y=klrfwd(net,x);
end
Y=reshape(y',npts,npts);
[c,h]=contour(X1,X2,Y,k,'r');
clabel(c, h);

hold on
plot(X(find(z==1),1),X(find(z==1),2),'+',X(find(z==0),1),X(find(z==0),2),'o');
%axis([minmax(1,1) minmax(1,2) minmax(2,1) minmax(2,2)]);
hold off
