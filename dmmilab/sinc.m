function y=sinc(x)
% SINC		-	computes the sinc function
% Y=SINC(X)
%
% (c) 1998 Robert F Harrison
% The University of Sheffield
%

tol=1.e-3;
y=zeros(size(x));
for i=1:length(x)
   z=abs(x(i));
   if z<tol
      y(i)=cos(z);
   else
      y(i)=sin(z)/z;
   end
end
