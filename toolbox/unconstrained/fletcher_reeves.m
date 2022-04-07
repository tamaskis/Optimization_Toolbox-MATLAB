%==========================================================================
%
% fletcher_reeves  Fletcher-Reeves update for the β parameter used by the
% conjugate gradient method.
%
%   beta = fletcher_reeves(g_curr,g_prev)
%
% Author: Tamas Kis
% Last Update: 2022-04-06
%
% REFERENCES:
%   [1] Kochenderfer and Wheeler, "Algorithms for Optimization" (p. 73)
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   g_curr  - (n×1 double) gradient of objective function at current
%             iteration
%   g_prev  - (n×1 double) gradient of objective function at previous
%             iteration
%
% -------
% OUTPUT:
% -------
%   beta    - (1×1 double) β for conjugate gradient method
%
%==========================================================================
function beta = fletcher_reeves(g_curr,g_prev)
    beta = (g_curr.'*g_curr)/(g_prev.'*g_prev);
end