clear;clc;close all;

addpath(genpath('toolbox'));

x_min_act = [2/3;1/sqrt(3)];

f = @(x) -x(1)*x(2)+2/(3*sqrt(3));
c = @(x) [ x(1)+x(2)^2-1;
          -x(1)-x(2)];
x0 = [2;2];
%x0 = [-10;-10];
x0 = x_min_act;
sigma0 = 10;
p = penalty_quadratic([],c,length(x0));

rho = 1e6;

f_pen = @(x) f(x)+rho*p(x);

opts.k_max = 20;
opts.m = 100;
opts.m_elite = 20;
x_min = minimize_cross_entropy(f_pen,x0,sigma0,opts);

% clear opts;
% opts.return_all = true;
% opts.alpha = 0.75;
% [x_min,~,~,x_all] = minimize_hooke_jeeves(f_pen,x0,opts);

clear opts;
opts.return_all = true;
opts.sigma0 = 1;
[x_min,~,k,x_all] = minimize_nelder_mead(f_pen,x0,opts);

%x_all

%feasible = check_feasibility(x_min,[],c)
k
x_min-x_min_act
