clear;
clc;
close all;

addpath(genpath('../..'))

f = @ (x) (x(1)-10)^4+(x(2)-40)^4;
x0 = [1;1];

%fmingrad(f,x0)

opts.k_max = 300;
fminbb(f,x0,opts)

opts.termination = 'rel';
opts.lambda = 0.6;
fminbb(f,x0,opts)


%opts.step_factor = 'decay';
%opts.alpha = 1;
%opts.gamma = 0.75;
%fmingrad(f,x0,opts)

%fminsearch(f,x0)

% f = @(x) x(1)^2+x(2)^2;
% x0 = [10;
%       10];
% 
% fmingrad(f,x0);
% 
% f = @(x) x^2;
%fminuni(f,10);