function dmroc(z,y)
% DMROC
% computes and plots ROC curve and its area.
%
% z is column vector of binary target values
% y is column vector of estimated probabilities

% (c) 2008 Robert F Harrison
% The University of Sheffield

nthresh=100;Np=sum(z);Nn=sum(1-z);
t=linspace(0,1,nthresh); % vector of threshold values
ptp=NaN*ones(size(t));pfa=ptp;ppv=ptp;npv=ptp; % initialize
ptp(1)=1;ptp(end)=0;pfa(1)=1;pfa(end)=0;ppv(1)=Np/(Np+Nn);ppv(end)=1;
for ii=2:nthresh-1
    tp=length(find(z>=t(ii) & y>=t(ii)));
    tn=length(find(z<t(ii) & y<t(ii)));
    fp=length(find(z<t(ii) & y>=t(ii)));
    fn=length(find(z>=t(ii) & y<t(ii)));
    ptp(ii)=tp/(tp+fn); % recall
    pfa(ii)=fp/(tn+fp);
    ppv(ii)=tp/(tp+fp); % precision
    npv(ii)=tn/(tn+fn);
end
area=vuroc(z,y);
%h=plot(pfa,ptp,'k',[0 1],[0 1],'k:',[0 0.5],[1 0.5],'k:');axis('square');
plot(pfa,ptp,'k',[0 1],[0 1],'k:',[0 0.5],[1 0.5],'k:');axis('square');
xlabel('False Positive Rate');ylabel('True Positive Rate');title('ROC Curve');
text(.5,.25,['AUROC = ' num2str(area,'%4.2f')]);


