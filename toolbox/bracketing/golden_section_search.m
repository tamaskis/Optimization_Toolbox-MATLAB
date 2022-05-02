%==========================================================================
%
% golden_section_search  Finds a bracketing interval containing a local 
% minimizer of a univariate objective function.
%
%   [a,b] = golden_section_search(f,a0,b0)
%   [a,b] = golden_section_search(f,a0,b0,n)
%   [a,b] = golden_section_search(f,a0,b0,[],TOL)
%
% Author: Tamas Kis
% Last Update: 2022-04-05
%
% REFERENCES:
%   [1] Kochenderfer and Wheeler, "Algorithms for Optimization", Algorithm
%       3.3 (p. 41)
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   f       - (1×1 function_handle) univariate objective function, f(x) 
%             (f : ℝ → ℝ)
%   a0      - (1×1 double) initial guess for lower bound of bracketing 
%             interval containing local minimizer of f(x)
%   b0      - (1×1 double) initial guess for upper bound of bracketing 
%             interval containing local minimizer of f(x)
%   n       - (1×1 double) (OPTIONAL) maximimum number of function 
%             evaluations
%               --> defaults to 100 if neither "n" nor "TOL" are input
%               --> determined using tolerance if "TOL" is input
%   TOL     - (1×1 double) (OPTIONAL) tolerance (only used if "n" is not
%             specified and you want to specify some tolerance)
%
% -------
% OUTPUT:
% -------
%   a       - (1×1 double) lower bound of bracketing interval containing 
%             local minimizer of f(x)
%   b       - (1×1 double) upper bound of bracketing interval containing 
%             local minimizer of f(x)
%
%==========================================================================
function [a,b] = golden_section_search(f,a0,b0,n,TOL)
    
    % ---------------------------------------------
    % Initializes parameters for Algorithm 3.3 [1].
    % ---------------------------------------------

    phi = (1+sqrt(5))/2;    % golden ratio
    a = a0;
    b = b0;

    % ------------------------------------------------
    % Defaults maximum number of function evaluations.
    % ------------------------------------------------
    
    % defaults to 100 if neither "n" nor "TOL" are input
    if (nargin < 4) || isempty(n)
        n = 100;
    end
    
    % defaults according to tolerance specification if TOL is input
    if (nargin == 5)
        n = ceil((b-a)/(TOL*log(phi)));
    end

    % ------------------------------------------
    % Algorithm 3.3 (golden section search) [1].
    % ------------------------------------------
    
    rho = phi-1;
    d = rho*b+(1-rho)*a;
    yd = f(d);
    for i = 1:(n-1)
        c = rho*a+(1-rho)*b;
        yc = f(c);
        if yc < yd
            b = d;
            d = c;
            yd = yc;
        else
            a = b;
            b = c;
        end
    end
      
end