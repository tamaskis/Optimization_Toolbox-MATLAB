%==========================================================================
%
% fminbb  Finds the local minimizer and minimum of an objective 
% function using the Barzilai-Borwein step factor.
%
%   x_min = fminbb(f,x0)
%   x_min = fminbb(f,x0,opts)
%   [x_min,f_min] = fminbb(__)
%   [x_min,f_min,x_all,f_all] = fminbb(__)
%
% Author: Tamas Kis
% Last Update: 2022-04-05
%
% REFERENCES:
%   [1] Kochenderfer and Wheeler, "Algorithms for Optimization" (pp. 69-71)
%
% This function requires the Numerical Differentiation Toolbox:
% https://www.mathworks.com/matlabcentral/fileexchange/97267-numerical-differentiation-toolbox
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   f       - (1×1 function_handle) objective function, f(x) (f : ℝⁿ → ℝ)
%   x0      - (n×1 double) initial guess for local minimizer
%   opts    - (1×1 struct) (OPTIONAL) solver options
%       • gradient      - (function_handle) gradient of the objective
%                         function
%       • k_max         - (1×1 double) maximimum number of iterations
%                         (defaults to 100)
%       • return_all    - (logical) all intermediate root estimates are
%                         returned if set to "true"; otherwise, a faster 
%                         algorithm is used to return only the converged 
%                         local minimizer/minimum
%       • termination   - (char) termination condition ('abs' or 'rel')
%       • TOL           - (1×1 double) tolerance (defaults to 1e-12)
%       • warnings      - (logical) true if any warnings should be
%                         displayed, false if not (defaults to true)
%
% -------
% OUTPUT:
% -------
%   x_min   - (n×1 double) local minimizer of f(x)
%   f_min   - (1×1 double) local minimum of f(x)
%   x_all   - (n×k double) all estimates of local minimizer of f(x)
%   f_all   - (1×k double) all estimates of local minimum of f(x)
%
% -----
% NOTE:
% -----
%   --> k = number of iterations it took for the solution to converge
%
%==========================================================================
function [x_min,f_min,x_all,f_all] = fminbb(f,x0,opts)
    
    % ----------------------------------
    % Sets (or defaults) solver options.
    % ----------------------------------
    
    % sets function handle for gradient (approximates using complex-step
    % approximation if not input)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'gradient')
        g = @(x) igradient(f,x);
    else
        g = @(x) opts.gradient(x);
    end
    
    % sets maximum number of iterations (defaults to 100)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'k_max')
        k_max = 100;
    else
        k_max = opts.k_max;
    end
    
    % sets parameter that scales the step factor
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'lambda')
        lambda = 1;
    else
        lambda = opts.lambda;
    end
    
    % determines if all intermediate estimates should be returned
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'return_all')
        return_all = false;
    else
        return_all = opts.return_all;
    end

    % sets termination condition (defaults to 'abs')
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'termination')
        condition = 'abs';
    else
        condition = opts.termination;
    end

    % sets tolerance (defaults to 1e-12)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'TOL')
        TOL = 1e-12;
    else
        TOL = opts.TOL;
    end
    
    % determines if warnings should be displayed (defaults to true)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'warnings')
        warnings = true;
    else
        warnings = opts.warnings;
    end

    % -----------------------------------------------------------
    % Gradient descent method using Barzilai-Borwein step factor.
    % -----------------------------------------------------------

    % preallocates arrays
    if return_all
        x_all = zeros(length(x0),k_max+1);
        f_all = zeros(1,k_max+1);
    end

    % sets 1st and 2nd estimates for local minimizer
    x_prev = x0;
    x_curr = x0+0.001;
    
    % gradient evaluation at 1st iteration
    g_prev = f(x_prev);
    
    % objective function evaluation at 2nd iteration
    f_curr = f(x_curr);

    % gradient descent method
    for k = 1:k_max
        
        % stores results in arrays
        if return_all
            x_all(:,k) = x_curr;
            f_all(k) = f_curr;
        end

        % gradient at current iteration
        g_curr = g(x_curr);

        % descent direction
        d = -g_curr/norm(g_curr);

        % Barzilai-Borwein step factor
        alpha = norm(g_curr)*(((x_curr-x_prev).'*(g_curr-g_prev))/...
            norm(g_curr-g_prev)^2);

        % scaling Barzilai-Borwein step factor
        alpha = lambda*alpha;

        % next estimate of local minimizer and minimum
        x_next = x_curr+alpha*d;
        f_next = f(x_next);

        % terminates solver if termination condition satisfied
        if terminate_solver(f_curr,f_next,TOL,condition)
            break;
        end

        % stores results/evaluations for next iteration
        g_prev = g_curr;
        x_prev= x_curr;
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
    
    % ---------------------------------------------------------
    % Displays warning if maximum number of iterations reached.
    % ---------------------------------------------------------

    if (k == k_max) && warnings
        warning(strcat('Maximum number of iterations (',num2str(k_max),...
            ') reached.'));
    end
      
end