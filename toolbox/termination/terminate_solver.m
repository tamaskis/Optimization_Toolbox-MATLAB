%==========================================================================
%
% terminate_solver  Terminates the optimization solver once a termination
% condition has been reached.
%
%   terminate = terminate_solver(f_curr,f_next,TOL,condition)
%
% Author: Tamas Kis
% Last Update: 2022-04-05
%
% REFERENCES:
%   [1] Kochenderfer and Wheeler, "Algorithms for Optimization" 
%       (pp. 63, 66)
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   f_curr      - (1×1 double) objective function evaluation at current
%                 design point, f(x⁽ᵏ⁾)
%   f_next      - (1×1 double) objective function evaluation at next
%                 design point, f(x⁽ᵏ⁺¹⁾)
%   TOL         - (1×1 double) tolerance
%   condition   - (char) specifies termination condition ('rel' for 
%                 relative, 'abs' for absolute)
%
% -----
% NOTE:
% -----
%                    | f(x⁽ᵏ⁾) - f(x⁽ᵏ⁺¹⁾) |
%   relative error = | ------------------- |
%                    |       f(x⁽ᵏ⁾)       |
%
%   absolute error = |f(x⁽ᵏ⁾) - f(x⁽ᵏ⁺¹⁾)|
%
%==========================================================================
function terminate = terminate_solver(f_curr,f_next,TOL,condition)
    if strcmpi(condition,'rel')
        terminate = (abs((f_curr-f_next)/f_curr) < TOL);
    else
        terminate = (abs(f_curr-f_next) < TOL);
    end
end