%==========================================================================
%
% fmingrad  Finds the local minimizer and minimum of an objective 
% function using the gradient descent method.
%
%   x_min = fmingrad(f,x0)
%   x_min = fmingrad(f,x0,opts)
%   [x_min,f_min] = fmingrad(__)
%   [x_min,f_min,k] = fmingrad(__)
%   [x_min,f_min,k,x_all,f_all] = fmingrad(__)
%
% Author: Tamas Kis
% Last Update: 2022-04-06
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
%       • alpha         - (1×1 double) learning rate (i.e. constant step
%                         factor)
%       • gamma         - (1×1 double) step factor decay rate
%       • gradient      - (function_handle) gradient of the objective
%                         function
%       • k_max         - (1×1 double) maximimum number of iterations
%                         (defaults to 200)
%       • return_all    - (logical) all intermediate root estimates are
%                         returned if set to "true"; otherwise, a faster 
%                         algorithm is used to return only the converged 
%                         local minimizer/minimum
%       • step_type     - (char) specifies type of step factor to use: 
%                         'line search', 'decay', or 'constant' (defaults
%                         to 'line search')
%                           --> if step_type = 'constant', then opts.alpha 
%                               must be specified
%                           --> if step_type = 'decay', then opts.alpha AND
%                               opts.gamma must be specified
%       • termination   - (char) termination condition ('abs' or 'rel')
%       • TOL           - (1×1 double) tolerance (defaults to 10⁻¹⁰)
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
function [x_min,f_min,k,x_all,f_all] = fmingrad_simple(f,x0,alpha)
    
    g = @(x) igradient(f,x);
    
    k_max = 13;
    TOL = 1e-10;


    % ------------------------
    % Gradient descent method.
    % ------------------------

    % preallocates arrays
    x_all = zeros(length(x0),k_max+1);
    f_all = zeros(1,k_max+1);

    % sets initial guess for local minimizer
    x_curr = x0;
    
    % objective function evaluation at initial guess
    f_curr = f(x_curr);

    % gradient descent method
    for k = 1:k_max
        
        % stores results in arrays
        x_all(:,k) = x_curr;
        f_all(k) = f_curr;

        % gradient at current iteration
        g_curr = g(x_curr);

        % descent direction
        d = -g_curr;

        % next estimate of local minimizer and minimum
        x_next = x_curr+alpha*d;
        f_next = f(x_next);

        % terminates solver if termination condition satisfied
        if terminate_solver(f_curr,f_next,TOL,'rel')
            break;
        end

        % stores variables for next iteration
        x_curr = x_next;
        f_curr = f_next;
        
    end

    % converged local minimizer and minimum
    x_min = x_next;
    f_min = f_next;

    % stores converged results and trims arrays
    x_all(:,k+1) = x_min; x_all = x_all(:,1:(k+1));
    f_all(k+1) = f_min; f_all = f_all(:,1:(k+1));
    
end