%==========================================================================
%
% minimize_hooke_jeeves  Finds the local minimizer and minimum of an 
% objective function using the Hooke-Jeeves method.
%
%   x_min = minimize_hooke_jeeves(f,x0,sigma0)
%   x_min = minimize_hooke_jeeves(f,x0,sigma0,opts)
%   [x_min,f_min] = minimize_hooke_jeeves(__)
%   [x_min,f_min,k] = minimize_hooke_jeeves(__)
%   [x_min,f_min,k,x_all,f_all] = minimize_hooke_jeeves(__)
%
% Author: Tamas Kis
% Last Update: 2022-05-02
%
% REFERENCES:
%   [1] Kochenderfer and Wheeler, "Algorithms for Optimization"
%       (pp. 102-104)
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   f       - (1×1 function_handle) objective function, f(x) (f : ℝⁿ → ℝ)
%   x0      - (n×1 double) initial guess for local minimizer
%   opts    - (1×1 struct) (OPTIONAL) solver options
%       • alpha      - (1×1 double) starting step size (defaults to 1)
%       • gamma      - (1×1 double) step size decay rate (defaults to 0.5)
%       • k_max      - (1×1 double) maximimum number of iterations
%                      (defaults to 200)
%       • return_all - (1×1 logical) all intermediate root estimates are
%                      returned if set to "true"; otherwise, a faster 
%                      algorithm is used to return only the converged local
%                      minimizer/minimum (defaults to false)
%       • TOL        - (1×1 double) tolerance (defaults to 10⁻¹⁰)
%
% -------
% OUTPUT:
% -------
%   x_min   - (n×1 double) local minimizer of f(x)
%   f_min   - (1×1 double) local minimum of f(x)
%   k       - (1×1 double) number of solver iterations
%   x_all   - (n×k double) all estimates of local minimizer of f(x)
%   f_all   - (1×k double) all estimates of local minimum of f(x)
%
%==========================================================================
function [x_min,f_min,k,x_all,f_all] = minimize_hooke_jeeves(f,x0,opts)
    
    % ----------------------------------
    % Sets (or defaults) solver options.
    % ----------------------------------
    
    % sets starting step size (defaults to 1)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'alpha')
        alpha = 1;
    else
        alpha = opts.alpha;
    end
    
    % sets step size decay rate (defaults to 0.5)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'gamma')
        gamma = 0.5;
    else
        gamma = opts.gamma;
    end
    
    % sets maximum number of iterations (defaults to 200)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'k_max')
        k_max = 200;
    else
        k_max = opts.k_max;
    end
    
    % determines if all intermediate estimates should be returned
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'return_all')
        return_all = false;
    else
        return_all = opts.return_all;
    end
    
    % sets tolerance (defaults to 10⁻¹⁰)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'TOL')
        TOL = 1e-10;
    else
        TOL = opts.TOL;
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
    x_best = x0;
    
    % objective function evaluation at initial guess
    f_best = f(x_best);
    
    % design point dimension
    n = length(x_best);
    
    % Hooke-Jeeves method
    for k = 1:k_max
        
        % stores results in arrays
        if return_all
            x_all(:,k) = x_best;
            f_all(k) = f_best;
        end
        
        % tracks if any improvement occurs
        improved = false;

        % current estimate for local minimizer
        x_curr = x_best;
        
        % steps in each direction, keeping track of any improvements
        for i = 1:n
            for sgn = -1:2:1
                x_step = x_curr+sgn*alpha*basis(i,n);
                f_step = f(x_step);
                if f_step < f_best
                    x_best = x_step;
                    f_best = f_step;
                    improved = true;
                end
            end
        end
        
        % decreases step size if no improvement
        if ~improved
            alpha = alpha*gamma;
        end
        
        % terminates if step size falls below tolerance
        if alpha < TOL
            break;
        end
        
    end
    
    % converged local minimizer and minimum
    x_min = x_best;
    f_min = f_best;
    
    % stores converged results and trims arrays
    if return_all
        x_all(:,k+1) = x_min; x_all = x_all(:,1:(k+1));
        f_all(k+1) = f_min; f_all = f_all(:,1:(k+1));
    end
    
end