%==========================================================================
%
% fmindescent  Finds the local minimizer and minimum of an objective 
% function using a descent method.
%
%   x_min = fmindescent(f,x0)
%   x_min = fmindescent(f,x0,opts)
%   [x_min,f_min] = fmindescent(__)
%
% Author: Tamas Kis
% Last Update: 2022-04-05
%
% REFERENCES:
%   [1] Kochenderfer and Wheeler, "Algorithms for Optimization" (pp. 53-54)
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
%       • i_max         - (1×1 double) maximimum number of iterations
%                         (defaults to 100)
%       • return_all    - (logical) all intermediate root estimates are
%                         returned if set to "true"; otherwise, a faster 
%                         algorithm is used to return only the converged 
%                         local minimizer/minimum
%       • TOL           - (1×1 double) tolerance (defaults to 1e-12)
%       • warnings      - (logical) true if any warnings should be
%                         displayed, false if not (defaults to true)
%
% -------
% OUTPUT:
% -------
%   x_min   - (n×1 or n×i double) local minimizer of f(x)
%           	--> if "return_all" is specified as "true", then "x_min"
%                   will be a matrix, where the first column is the initial
%                   guess, the last column is the converged local 
%                   minimizer, and the other columns are intermediate 
%                   estimates of the local minimizer
%               --> otherwise, "x_min" is a single column vector storing
%                   the converged local minimizer
%   f_min   - (1×1 or 1×i double) local minimum of f(x)
%           	--> if "return_all" is specified as "true", then "f_min" 
%                   will be a vector, where the first element is the 
%                   initial guess, the last element is the converged local 
%                   minimum, and the other elements are intermediate 
%                   estimates of the local minimum
%               --> otherwise, "f_min" is a scalar storing the converged
%                   local minimum
%
% -----
% NOTE:
% -----
%   --> "i" is the number of iterations it took for the solution to
%       converge.
%
%==========================================================================
function [x_min,f_min] = fmindescent(f,x0,opts)
    
    % ----------------------------------
    % Sets (or defaults) solver options.
    % ----------------------------------

    % sets function handle for gradient (approximates using complex-step
    % approximation if not input)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'gradient')
        grad = @(x) igradient(f,x);
    else
        grad = @(x) opts.gradient(x);
    end

    % sets maximum number of iterations (defaults to 100)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'i_max')
        i_max = 100;
    else
        i_max = opts.i_max;
    end
    
    % determines return value (defaults to only return converged values)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'return_all')
        return_all = false;
    else
        return_all = opts.return_all;
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
    % "Return all" implementation of the gradient descent method.
    % -----------------------------------------------------------
    
    if return_all
        
        % dimension of vector variable
        n = length(x0);
        
        % preallocates arrays to store intermediate local minimizer 
        % estimates (x) and gradients (g)
        x = zeros(n,i_max);
        g = zeros(n,i_max);
        
        % inputs 1st and 2nd guesses for local minimizer into x vector
        x(:,1) = x0;
        x(:,2) = x0+0.001;

        % gradient for first iteration
        g(:,1) = grad(x0);
        
        % initializes the error so the loop will be entered
        err = 2*TOL;

        % gradient descent method
        i = 2;
        while (err > TOL) && (i < i_max)
            
            % calculates gradient
            g(:,i) = grad(x(:,i));
            
            % calculates learning rate
            gamma = lambda*abs((x(:,i)-x(:,i-1))'*(g(:,i)-g(:,i-1)))/...
                norm(g(:,i)-g(:,i-1))^2;

            % updates estimate of local minimizer
            x(:,i+1) = x(:,i)-gamma*g(:,i);
    
            % calculates error
            err = norm(x(:,i+1)-x(:,i));
    
            % increments loop index
            i = i+1;
           
        end
        
        % returns converged local minimizer and minimum along with their 
        % intermediate estimates
        x_min = x(:,1:i);
        f_min = zeros(i,1);
        for j = 1:i
            f_min(j) = f(x(:,j));
        end

    % -----------------------------------------------------
    % "Fast" implementation of the gradient descent method.
    % -----------------------------------------------------

    else

        % sets 1st and 2nd estimates for local minimizer
        x_old = x0;
        x_int = x0+0.001;

        % gradient for the initial guess
        g_old = grad(x0);
        
        % initializes x_new so its scope isn't limited to the while loop
        x_new = zeros(size(x0));
        
        % initializes the error so the loop will be entered
        err = 2*TOL;

        % gradient descent method
        i = 2;
        while (err > TOL) && (i < i_max)
            
            % gradient at current iteration
            g_int = grad(x_int);
            
            % calculates learning rate
            gamma = lambda*abs((x_int-x_old)'*(g_int-g_old))/...
                norm(g_int-g_old)^2;

            % updates estimate of local minimizer
            x_new = x_int-gamma*g_int;

            % calculates error
            err = norm(x_new-x_int);
            
            % stores current and previous local minimizer estimates for 
            % next iteration
            x_old = x_int;
            x_int = x_new;

            % stores gradient for next iteration
            g_old = g_int;

            % increments loop index
            i = i+1;

        end

        % returns converged local minimizer and minimum
        x_min = x_new;
        f_min = f(x_new);

    end

    % ---------------------------------------------------------
    % Displays warning if maximum number of iterations reached.
    % ---------------------------------------------------------

    if (i == i_max) && warnings
        warning(strcat('The method failed after i=',num2str(i_max),...
            ' iterations.'));
    end
      
end