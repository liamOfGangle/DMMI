function opt_vect = foptionsrfh()
% FOPTIONS Sets default parameters for optimisation routines
% For compatibility with MATLAB's foptions()
%
% Copyright (c) Dharmesh Maniyar, Ian T. Nabney (2004)
  
opt_vect      = zeros(1, 18);
opt_vect(2:3) = 1e-2;
opt_vect(4)   = 1e-2;
opt_vec(14)   = 100;
opt_vect(15)  = 1e-3;
opt_vect(16)  = 1e-8;
opt_vect(17)  = 0.1;