function [y_hat,y_star_hat,w_hat,Psi,Psi_star]=grbfnn(Z,X,X_star,C,ell,rho)
% [y_hat,y_star_hat,w_hat,Psi,Psi_star]=grbfnn(Z,X,X_star,C,ell,rho)
% computes a gaussian radial basis function nn for fixed centres,C and
% widths, ell, and an optional L_2 regularisation parameter, rho
%
% It returns sample and oos outputs, y_hat & y_hat_star, and
% also the optimal weights, w_hat, & the basis functions located at C,
% Psi & Psi_star
%
if nargin<6;rho=0;end
n=size(X,1);n_star=size(X_star,1);
m=size(C,1)+1;
Psi=[ones(n,1) exp(-dist2(X,C)/2/(ell^2))]; % construct design matrices
Psi_star=[ones(n_star,1) exp(-dist2(X_star,C)/2/(ell^2))];

%w_hat=inv(Psi'*Psi)*Psi'*Z; % compute weights
I=eye(m);I(1,1)=0; % exclude bias from regularization
w_hat=(Psi'*Psi+rho*I)\Psi'*Z; % compute weights

y_hat=Psi*w_hat;
y_star_hat=Psi_star*w_hat;

