%==========================================================================
%
% sample_statistics  Mean and covariance of a sample.
%
%   [mu,Sigma] = sample_statistics(x)
%
% Author: Tamas Kis
% Last Update: 2022-05-01
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   x       - (n×N double) N samples of X
%
% -------
% OUTPUT:
% -------
%   mu      - (n×1 double) mean of X
%   Sigma   - (n×n double) covariance of X
%
%==========================================================================
function [mu,Sigma] = sample_statistics(x)
    
    % dimension of X
    n = size(x,1);
    
    % sample size
    N = size(x,2);
    
    % mean
    mu = zeros(n,1);
    for i = 1:N
        mu = mu+x(:,i);
    end
    mu = mu/N;
    
    % covariance
    Sigma = zeros(n,n);
    for i = 1:N
        Sigma = Sigma+(x(:,i)-mu)*(x(:,i)-mu).';
    end
    Sigma = Sigma/(N-1);
    
end