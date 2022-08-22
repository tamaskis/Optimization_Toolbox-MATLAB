%==========================================================================
%
% minimize_cross_entropy  Finds the local minimizer and minimum of an 
% objective function using the cross entropy method.
%
%   x_min = minimize_cross_entropy(f,x0)
%   x_min = minimize_cross_entropy(f,x0,opts)
%   [x_min,f_min] = minimize_cross_entropy(__)
%   [x_min,f_min,x_all,f_all] = minimize_cross_entropy(__)
%
% Author: Tamas Kis
% Last Update: 2022-05-23
%
% REFERENCES:
%   [1] Kochenderfer and Wheeler, "Algorithms for Optimization"
%       (pp. 133-137)
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   f       - (1×1 function_handle) objective function, f(x) (f : ℝⁿ → ℝ)
%   x0      - (n×1 double) initial guess for local minimizer
%   sigma0  - (1×1 double) standard deviation of initial proposal
%             distribution (same for all design variables)
%   opts    - (1×1 struct) (OPTIONAL) solver options
%       • k_max      - (1×1 double) maximimum number of iterations 
%                      (defaults to 200)
%       • m          - (1×1 double) sample size
%       • m_elite    - (1×1 double) number of elite samples
%       • return_all - (logical) all intermediate root estimates are
%                      returned if set to "true"; otherwise, a faster 
%                      algorithm is used to return only the converged local
%                      minimizer/minimum
%       • sigma0     - (1×1 double) standard deviation of initial proposal
%                      distribution (same for all design variables)
%                      (defaults to 10)
%
% -------
% OUTPUT:
% -------
%   x_min   - (n×1 double) local minimizer of f(x)
%   f_min   - (1×1 double) local minimum of f(x)
%   x_all   - (n×k double) all estimates of local minimizer of f(x)
%   f_all   - (1×k double) all estimates of local minimum of f(x)
%
%==========================================================================
function [x_min,f_min,x_all,f_all] = minimize_cross_entropy(f,x0,opts)
    
    % ----------------------------------
    % Sets (or defaults) solver options.
    % ----------------------------------
    
    % sets maximum number of iterations (defaults to 200)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'k_max')
        k_max = 200;
    else
        k_max = opts.k_max;
    end
    
    % determines sample size
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'m')
        m = 100;
    else
        m = opts.m;
    end
    
    % determines number of elite samples
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'m_elite')
        m_elite = 10;
    else
        m_elite = opts.m_elite;
    end
    
    % determines if all intermediate estimates should be returned
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'return_all')
        return_all = false;
    else
        return_all = opts.return_all;
    end
    
    % sets proposal distribution standard deviation (defaults to 10)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'sigma0')
        sigma0 = 10;
    else
        sigma0 = opts.sigma0;
    end

    % ---------------------
    % Optimization routine.
    % ---------------------
    
    % preallocates arrays
    if return_all
        x_all = zeros(length(x0),k_max+1);
        f_all = zeros(1,k_max+1);
    end
    
    % sets initial guess for local minimizer
    x_curr = x0;
    
    % objective function evaluation at initial guess
    f_curr = f(x_curr);
    
    % specifies initial proposal distribution (Gaussian)
    mu = x_curr;
    Sigma = diag(sigma0^2*ones(1,length(mu)));
    
    % cross entropy method
    for k = 1:k_max
        
        % stores results in arrays
        if return_all
            x_all(:,k) = x_curr;
            f_all(k) = f_curr;
        end
        
        % samples from multivariate normal distribution (occasionally, the
        % covariance will get very small numerically, so we use only the
        % diagonal elements to ensure positive semidefiniteness.
        try
            x = randmvn(mu,Sigma,m);
        catch
            x = randmvn(mu,diag(diag(Sigma)),m);
        end
        
        % function evaluations for each sample
        f_eval = zeros(m,1);
        for i = 1:m
            f_eval(i) = f(x(:,i));
        end
        
        % indices of the smallest to largest elements of f_eval
        [~,idx] = sort(f_eval);
        
        % sorts the columns of x
        x = x(:,idx);
        
        % keeps only the top "m_elite" samples
        x = x(:,1:m_elite);
        
        % calculates new mean and covariance
        [mu,Sigma] = sample_statistics(x);
        
        % next estimate of local minimizer and minimum
        x_next = mu;
        f_next = f(x_next);
        
        % stores variables for next iteration
        x_curr = x_next;
        f_curr = f_next;
        
    end
    
    % converged local minimizer and minimum
    x_min = x_next;
    f_min = f_next;
    
    % stores converged results and trims arrays
    if return_all
        x_all(:,k+1) = x_min; x_all = x_all(:,1:(k+1));
        f_all(k+1) = f_min; f_all = f_all(:,1:(k+1));
    end
    
end