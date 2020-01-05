function [net, options, cost] = mlptrain(net,options,x,t);
%MLPTRAIN utility to train an MLP network
%
%	Description
%
%	[NET, OPTIONS, COST] = MLPTRAIN(NET, OPTIONS, X, T) trains a network data
%	structure NET using the scaled conjugate gradient algorithm  with input
%   data, X, target data, T.
%   Modify this file to train with different algorithms
%   alg = {'conjgrad', 'quasinew', 'scg', 'gradesc', 'olgd' }.
%
%	See also
%	NETOPT
%

%	Copyright (c) Ian T Nabney (1996-2001)
%   Modified R F Harrison 2007. Overloads the original NetLab
%   implementation
alg = {'conjgrad', 'quasinew', 'scg', 'gradesc', 'olgd' };
[net, options, cost] = netopt(net, options, x, t, alg{3});

