%==========================================================================
%
% line_search  Obtains the step factor for a descent method using line
% search.
%
%   alpha = line_search(f,x,d)
%   alpha = line_search(f,x,d,opts)
%
% Author: Tamas Kis
% Last Update: 2022-04-05
%
% REFERENCES:
%   [1] Kochenderfer and Wheeler, "Algorithms for Optimization" (p. 54)
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   f       - (1×1 function_handle) objective function, f(x) (f : ℝⁿ → ℝ)
%   x       - (n×1 double) design point
%   d       - (n×1 double) descent direction
%   opts    - (1×1 struct) (OPTIONAL) solver options for univariate
%             optimization (see "unconstrained/fminuni" for full 
%             definition)
%
% -------
% OUTPUT:
% -------
%   alpha   - (1×1 double) step factor, α
%
%==========================================================================
function alpha = line_search(f,x,d,opts)
    
    % objective function for line search
    g = @(alpha) f(x+alpha*d);

    % interval containing local minimizer of g(α)
    [a,b] = bracket_minimum(g);
    
    % initial guess for local minimizer of g(α)
    alpha0 = (a+b)/2;
    
    % finds step factor, α
    if (nargin == 4) && ~isempty(opts)
        alpha = fminuni(g,alpha0,opts);
    else
        alpha = fminuni(g,alpha0);
    end

end