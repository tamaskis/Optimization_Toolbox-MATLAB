%==========================================================================
%
% bracket_sign_change  Finds an interval in which a sign change occurs.
%
%   [a,b] = bracket_sign_change(f,a0,b0)
%
% Author: Tamas Kis
% Last Update: 2022-04-05
%
% REFERENCES:
%   [1] Kochenderfer and Wheeler, "Algorithms for Optimization", Algorithm
%       3.7 (p. 50)
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   f       - (1×1 function_handle) univariate function, f(x) (f : ℝ → ℝ)
%   a0      - (1×1 double) initial guess for lower bound of interval
%             containing sign change
%   b0      - (1×1 double) initial guess for upper bound of interval
%             containing sign change
%
% -------
% OUTPUT:
% -------
%   a       - (1×1 double) lower bound of interval containing sign change
%   b       - (1×1 double) lower bound of interval containing sign change
%
%==========================================================================
function [a,b] = bracket_sign_change(f,a0,b0)

    % ---------------------------------------------
    % Initializes parameters for Algorithm 3.7 [1].
    % ---------------------------------------------
    
    a = a0;
    b = b0;
    k = 2;

    % ----------------------------------------
    % Algorithm 3.7 (bracket sign change) [1].
    % ----------------------------------------

    if (a > b)
        a_old = a;
        b_old = b;
        a = b_old;
        b = a_old;
    end
    center = (b+a)/2;
    half_width = (b-a)/2;
    while (f(a)*f(b) > 0)
        half_width = half_width*k;
        a = center-half_width;
        b = center+half_width;
    end

end