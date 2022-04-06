%==========================================================================
%
% polak_ribiere  Polak-Ribière update for the β parameter used by the
% conjugate gradient method.
%
%   beta = polak_ribiere(g,g_prev)
%
% Author: Tamas Kis
% Last Update: 2022-04-05
%
% REFERENCES:
%   [1] Kochenderfer and Wheeler, "Algorithms for Optimization" (p. 73)
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   g       - (n×1 double) gradient of objective function at current
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
function beta = polak_ribiere(g,g_prev)
    beta = max((g.'*(g-g_prev))/(g_prev.'*g_prev),0);
end