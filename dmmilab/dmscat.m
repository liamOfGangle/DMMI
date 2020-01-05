function dmscat(z1,y1,z2,y2)
%
% DMSCAT creates scatter plots of data series overlaid with least-squares
% fit and estimates of correlation (r) and significance (p) (p<0.05 implies significance).
% z denotes measured (true) outcome
% y denotes estimated outcome
%
% Usage:
%   dmscat(z1,y1) is useful for comparing training (1) OR
%   validation runs
%   dmscatter(z1,y1,z2,y2) is useful for comparing training (1) AND
%   validation (2) runs

% (c) 2008 Robert F Harrison
% The University of Sheffield

[nz,mz]=size(z1);[ny,my]=size(y1);
if mz>1 | my>1;error('only column vectors permitted');end
if nz~=ny;error('vectors must be same length');end
if nargin >2
    if nargin ~=4;error('only two or four arguments permitted');
    else
        [nz,mz]=size(z2);[ny,my]=size(y2);
        if mz>1 | my>1;error('only column vectors permitted');end
        if nz~=ny;error('vectors must be same length');end
    end
end

sw=strcmp(get(0,'defaultFigureInvertHardcopy'),'off');

if sw==1
    col='w';
else
    col='k';
end

if nargin == 2
    m1=min(min([z1,y1]));
    m2=max(max([z1,y1]));
    plot(z1,y1,'color',col,'linestyle','none','marker','o','markerfacecolor',col);
    line([m1 m2 ],[m1 m2],'color',col);
    xlabel('measured');ylabel('estimated');
    [c,p]=corrcoef(z1,y1);c=c(2,1);p=p(2,1);
    v=axis;tx=v(1)+0.2*(v(2)-v(1));ty=v(3)+0.1*(v(4)-v(3));
    text(tx,ty,['corr. coeff. = ' num2str(c,'%4.2f')])
else
    m1=min(min([z1;y1]),min([z2;y2]));
    m2=max(max([z1;y1]),max([z2;y2]));
    plot(z1,y1,z2,y2,'color',col,'linestyle','none','marker','+','markerfacecolor',col);
    xlabel('measured');ylabel('estimated');
    line([m1 m2 ],[m1 m2],'color',col);
    v=axis;tx=v(1)+0.2*(v(2)-v(1));ty=v(3)+0.1*(v(4)-v(3));
    [c1,p1]=corrcoef(z1,y1);c1=c1(2,1);p1=p1(2,1);
    [c2,p2]=corrcoef(z2,y2);c2=c2(2,1);p2=p2(2,1);
    text(tx,ty,['corr. coeff. (1) = ' num2str(c1,'%4.2f') ' corr. coeff. (2) = ' num2str(c2,'%4.2f')])
end



