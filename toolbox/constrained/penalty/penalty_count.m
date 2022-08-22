%==========================================================================
%
% penalty_count  Defines a count penalty function.
%
%   p = penalty_count(h,g)
%   p = penalty_count(h,[])
%   p = penalty_count([],g)
%
% Author: Tamas Kis
% Last Update: 2022-05-02
%
% REFERENCES:
%   [1] Kochenderfer and Wheeler, "Algorithms for Optimization"
%       (p. 178)
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   h       - (1×1 function_handle) (OPTIONAL) function defining the 
%             equality constraint h(x) = 0 (h : ℝⁿ → ℝᵐ)
%   g       - (1×1 function_handle) (OPTIONAL) function defining the 
%             inequality constraint g(x) ≤ 0 (g : ℝⁿ → ℝˡ)
%
% -------
% OUTPUT:
% -------
%   p       - (1×1 function_handle) count penalty function, p(x) 
%             (p : ℝⁿ → ℝ)
%
%==========================================================================
function p = penalty_count(h,g)
    
    % defaults equality constraint to 0 if not input
    if isempty(h)
        h = @(x) 0;
    end
    
    % defaults inequality constraint to 0 if not input
    if isempty(g)
        g = @(x) 0;
    end
    
    % defines count penalty function
    p = @(x) sum(g(x)>0)+sum(h(x)~=0);
    
end