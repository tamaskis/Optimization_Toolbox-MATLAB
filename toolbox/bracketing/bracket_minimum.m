%==========================================================================
%
% bracket_minimum  Finds an interval in which a local minimum must exist.
%
%   [a,b] = bracket_minimum(f)
%   [a,b] = bracket_minimum(f,x0,s,k)
%
% Author: Tamas Kis
% Last Update: 2022-04-05
%
% REFERENCES:
%   [1] Kochenderfer and Wheeler, "Algorithms for Optimization", Algorithm
%       3.1 (p. 36)
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   f       - (1×1 function_handle) univariate objective function, f(x) 
%             (f : ℝ → ℝ)
%   x0      - (1×1 double) (OPTIONAL) initial guess for local minimizer
%             (defaults to 0)
%   s0      - (1×1 double) (OPTIONAL) initial step size (defaults to 0.01)
%   k0      - (1×1 double) (OPTIONAL) initial expansion factor (defaults to 
%             2)
%
% -------
% OUTPUT:
% -------
%   a       - (1×1 double) lower bound of interval containing local minimum
%   b       - (1×1 double) upper bound of interval containing local minimum
%
%==========================================================================
function [a,b] = bracket_minimum(f,x0,s0,k0)
    
    % -----------------------------
    % Defaults optional parameters.
    % -----------------------------

    % defaults initial guess for local minimizer to 0 if not input
    if (nargin < 2) || isempty(x0)
        x0 = 0;
    end

    % defaults initial step size to 0.01 if not input
    if (nargin < 3) || isempty(s0)
        s0 = 0.01;
    end

    % defaults initial expansion factor to 2 if not input
    if (nargin < 4) || isempty(k0)
        k0 = 2;
    end

    % ---------------------------------------------
    % Initializes parameters for Algorithm 3.1 [1].
    % ---------------------------------------------

    x = x0;
    s = s0;
    k = k0;
    
    % ------------------------------------
    % Algorithm 3.1 (bracket minimum) [1].
    % ------------------------------------
    
    a = x;
    ya = f(x);
    b = a+s;
    yb = f(b);
    if yb > ya
        a_old = a;
        b_old = b;
        a = b_old;
        b = a_old;
        yb = ya;
        s = -s;
    end
    while true
        c = b+s;
        yc = f(c);
        if yc > yb
            if a < c
                b = c;
                return
            else
                b = a;
                a = c;
                return;
            end
        end
        a = b;
        b = c;
        yb = yc;
        s = s*k;
    end
      
end