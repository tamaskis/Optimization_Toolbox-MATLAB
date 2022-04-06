%==========================================================================
%
% fminuni  Finds the local minimizer and minimum of a univariate 
% objective function (unconstrained optimization).
%
%   x_min = fminuni(f,x0)
%   x_min = fminuni(f,x0,opts)
%   [x_min,f_min] = fminuni(__)
%
% Author: Tamas Kis
% Last Update: 2022-04-05
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   f       - (1×1 function_handle) univariate objective function, f(x) 
%             (f : ℝ → ℝ)
%   x0      - (1×1 double) initial guess for local minimizer
%   opts    - (1×1 struct) (OPTIONAL) solver options
%       • method - (char) bracketing method: 'golden' (golden section
%                  search)
%                   --> defaults to 'golden'
%       • n      - (1×1 double) maximimum number of function evaluations 
%                  (defaults to 100)
%       • TOL    - (1×1 double) tolerance (TODO) (defaults to 1e-12)
%
% -------
% OUTPUT:
% -------
%   x_min   - (1×1 double) local minimizer of f(x)
%   f_min   - (1×1 double) local minimum of f(x)
%
%==========================================================================
function [x_min,f_min] = fminuni(f,x0,opts)
    
    % ----------------------------------
    % Sets (or defaults) solver options.
    % ----------------------------------
    
    % sets bracketing method (defaults to golden section search)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'method')
        method = 'golden';
    else
        method = opts.method;
    end
    
    % sets maximum number of function evaluations defaults to 100)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'n')
        n = 100;
    else
        n = opts.n;
    end
    
    % -------------
    % Optimization.
    % -------------
    
    % finds an initial interval containing local minimizer
    [a0,b0] = bracket_minimum(f,x0);
    
    % finds a (very small) interval containing local minimizer
    if strcmpi(method,'golden')
        [a,b] = golden_section_search(f,a0,b0,n);
    end
    
    % local minimizer and minimum
    x_min = (a+b)/2;
    f_min = f(x_min);
    
end