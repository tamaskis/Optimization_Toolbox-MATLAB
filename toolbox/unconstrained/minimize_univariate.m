%==========================================================================
%
% minimize_univariate  Finds the local minimizer and minimum of a 
% univariate objective function (unconstrained optimization).
%
%   x_min = minimize_univariate(f,x0)
%   x_min = minimize_univariate(f,x0,opts)
%   [x_min,f_min] = minimize_univariate(__)
%
% Author: Tamas Kis
% Last Update: 2022-04-06
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
%                  (defaults to 200)
%       • TOL    - (1×1 double) tolerance (defaults to 10⁻¹⁰)
%
% -------
% OUTPUT:
% -------
%   x_min   - (1×1 double) local minimizer of f(x)
%   f_min   - (1×1 double) local minimum of f(x)
%
%==========================================================================
function [x_min,f_min] = minimize_univariate(f,x0,opts)
    
    % ----------------------------------
    % Sets (or defaults) solver options.
    % ----------------------------------
    
    % sets bracketing method (defaults to golden section search)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'method')
        method = 'golden';
    else
        method = opts.method;
    end
    
    % sets maximum number of function evaluations (defaults to 200)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'n')
        n = 200;
    else
        n = opts.n;
    end
    
    % determines if tolerance is input
    TOL_input = (nargin == 3) && ~isempty(opts) && isfield(opts,'TOL');
    
    % -------------
    % Optimization.
    % -------------
    
    % finds an initial interval containing local minimizer
    [a0,b0] = bracket_minimum(f,x0);
    
    % finds a (very small) interval containing local minimizer
    if strcmpi(method,'golden')
        if TOL_input
            [a,b] = golden_section_search(f,a0,b0,n,TOL);
        else
            [a,b] = golden_section_search(f,a0,b0,n);
        end
    end
    
    % local minimizer and minimum
    x_min = (a+b)/2;
    f_min = f(x_min);
    
end