function [e, edata, eprior, y, a] = rbferr(net, x, t)
%RBFERR	Evaluate error function for RBF network.
%
%	Description
%	E = RBFERR(NET, X, T) takes a network data structure NET together
%	with a matrix X of input vectors and a matrix T of target vectors,
%	and evaluates the appropriate error function E depending on
%	NET.OUTFN.  Each row of X corresponds to one input vector and each
%	row of T contains the corresponding target vector.
%
%	[E, EDATA, EPRIOR] = RBFERR(NET, X, T) additionally returns the data
%	and prior components of the error, assuming a zero mean Gaussian
%	prior on the weights with inverse variance parameters ALPHA and BETA
%	taken from the network data structure NET.
%
%	See also
%	RBF, RBFFWD, RBFGRAD, RBFPAK, RBFTRAIN, RBFUNPAK
%

%	Copyright (c) Ian T Nabney (1996-2001)
%   Modified to accept logistic and softmax output functions
%   R F Harrison (2003)


% Check arguments for consistency
switch net.outfn
case 'linear'
   errstring = consist(net, 'rbf', x, t);
case 'logistic'
   errstring = consist(net, 'rbf', x, t);
case 'softmax'
   errstring = consist(net, 'rbf', x, t);
case 'neuroscale'
   errstring = consist(net, 'rbf', x);
otherwise
   error(['Unknown output function ', net.outfn]);
end
if ~isempty(errstring);
  error(errstring);
end
[y,z,n2,a] = rbffwd(net, x); % changed by RFH to emulate glmerr
switch net.outfn
case 'linear'
   edata = 0.5*sum(sum((y - t).^2));
case 'logistic'
   edata = - sum(sum(t.*log(y) + (1 - t).*log(1 - y)));
case 'softmax'
   edata = - sum(sum(t.*log(y)));
case 'neuroscale'
   y = rbffwd(net, x);
   y_dist = sqrt(dist2(y, y));
   % Take t as target distance matrix
   edata = 0.5.*(sum(sum((t-y_dist).^2)));
otherwise
   error(['Unknown output function ', net.outfn]);
end

% Compute Bayesian regularised error
[e, edata, eprior] = errbayes(net, edata);

