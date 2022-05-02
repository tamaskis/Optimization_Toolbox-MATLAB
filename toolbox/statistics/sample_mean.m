%==========================================================================
%
% sample_mean  Mean of a sample.
%
%   mu = sample_mean(x)
%
% Author: Tamas Kis
% Last Update: 2022-05-02
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
%
%==========================================================================
function mu = sample_mean(x)
    
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
    
end