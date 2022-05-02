clear;
clc;
close all;

addpath(genpath('../..'))


n = 200;
x0 = [-5;10];
f = @(x) 100*(x(2)-x(1)^2)^2+(1-x(1))^2;

%n = 40;
%x0 = [5,5]';
%f = @(x) (x(1)^2+x(2)-11)^2+(x(1)+x(2)^2-7)^2;

% n = 100;
% x0 = [100;500;10;10];
% f = @(x) (x(1)+10*x(2))^2+5*(x(3)-x(4))^2+(x(2)-2*x(3))^4+10*(x(1)-x(4))^4;

opts.k_max = floor(n/3);
opts.lambda = 1;
[x_min,f_min,k] = fminbb(f,x0,opts)
%[x_min,~,~,output] = fminunc(f,x0)

% 
% clear opts;
% opts.alpha = 10;
% opts.step_type = 'constant';
% opts.return_all = true;
% opts.k_max = 10;
% [x_min,f_min,k] = fmingrad(f,x0,opts);
% [x_min,f_min,k] = fmingrad_simple(f,x0,0.01)


% n = 100;
% opts.k_max = floor(n/3);
% opts.lambda = 1;
% f = @(x) (x(1)+10*x(2))^2+5*(x(3)-x(4))^2+(x(2)-2*x(3))^4+10*(x(1)-x(4))^4;
% x0 = [5;5;5;5];
% [x_min,f_min] = fminbb(f,x0,opts)




% x0 = [1.01;1.01];
% 
% opts.k_max = 6;
% opts.alpha = 10;
% opts.gamma = 0.5;
% opts.step_factor = 'decay';
% fmingrad(f,x0,opts)
% 
% clear opts;
% 
% opts.k_max = 6;
% opts.alpha = 10;
% opts.gamma = 0.5011;
% opts.step_factor = 'decay';
% x_min = fmingrad(f,x0,opts);
% norm([1;1]-x_min)
