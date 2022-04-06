clear;
clc;
close all;

f = @ (x) (x(1)-10)^4+(x(2)-40)^10;
x0 = [1;1];

fmingrad(f,x0)

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