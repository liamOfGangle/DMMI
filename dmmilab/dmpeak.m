function dmpeak(x,z,x_star,y_star_hat,y_true)
% dmpeak(x,z,x_star,y_star_hat,y_true)
% is a helper to plot your 'peaks' data
% y_true is assumed to corespond to the points in x_star
%
n=sqrt(size(x,1));
n_star=sqrt(size(x_star,1));
X1=reshape(x(:,1),n,n);
X2=reshape(x(:,2),n,n);
X1_star=reshape(x_star(:,1),n_star,n_star);
X2_star=reshape(x_star(:,2),n_star,n_star);
Y_star_hat=reshape(y_star_hat,n_star,n_star);
Z=reshape(z,n,n);
Y_true=reshape(y_true,n_star,n_star);

subplot(121);surf(X1_star,X2_star,Y_star_hat);axis([-5,5,-5,5,-10,10])
xlabel('input_1');ylabel('input_2');zlabel('output');title('estimate')
hold on
stem3(X1,X2,Z);axis([-5,5,-5,5,-10,10])
hold off

subplot(122);surf(X1_star,X2_star,Y_true);axis([-5,5,-5,5,-10,10])
xlabel('input_1');ylabel('input_2');zlabel('output');title('true')
hold on
stem3(X1,X2,Z);axis([-5,5,-5,5,-10,10])
hold off




