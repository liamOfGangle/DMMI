function dmplotres(z,y)
% plotres(z,y)
% plots scalar residuals after regression analysis
% z - observed value (target)
% y - predicted value
% NB incorrect ordering of arguments will lead to misleading results
%
[y,ind]=sort(y);z=z(ind); % order data on fitted value (min to max)
if exist('DM','var')
    plot(y,z-y,'wo','markerfacecolor','w');
else
    plot(y,z-y,'ko','markerfacecolor','k');
end
axis('tight')
v=axis;axis([v(1) v(2) 1.5*v(3) 1.5*v(4)])
m=mean(z-y);
s=std(z-y);
v=axis;tx=v(1)+0.1*(v(2)-v(1));ty=v(3)+0.95*(v(4)-v(3));

text(tx,ty,['std. dev. = ' num2str(s,'%4.2f')])
ylabel('residual');xlabel('predicted value')