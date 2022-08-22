%==========================================================================
%
% check_feasibility  Checks the feasiblity of a design point.
%
%   feasible = check_feasibility(x,h,g)
%
% Author: Tamas Kis
% Last Update: 2022-05-01
%
% REFERENCES:
%   [1] Kochenderfer and Wheeler, "Algorithms for Optimization"
%       (pp. 168-169)
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   x           - (n×1 double) design point
%   h           - (1×1 function_handle) (OPTIONAL) function defining the 
%                 equality constraint h(x) = 0 (h : ℝⁿ → ℝᵐ)
%   g           - (1×1 function_handle) (OPTIONAL) function defining the 
%                 inequality constraint g(x) ≤ 0 (g : ℝⁿ → ℝˡ)
%
% -------
% OUTPUT:
% -------
%   feasible    - (1×1 logical) "true" if design point is feasible, "false"
%                 otherwise
%
%==========================================================================
function feasible = check_feasibility(x,h,g)
    
    % defaults equality constraint to 0 if not input
    if isempty(h)
        h = @(x) 0;
    end
    
    % defaults inequality constraint to 0 if not input
    if isempty(g)
        g = @(x) 0;
    end
    
    % evaluates contraints
    h_eval = h(x);
    g_eval = g(x);
    
    % checks feasibility
    feasible = (sum(h_eval == zeros(size(h_eval)))) &&...
        (sum(g_eval <= zeros(size(g_eval))));
    
end