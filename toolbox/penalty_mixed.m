%==========================================================================
%
% penalty_mixed  Defines a mixed penalty function.
%
%   p = penalty_mixed(h,g,rho1,rho2)
%   p = penalty_mixed(h,[],rho1,rho2)
%   p = penalty_mixed([],g,rho1,rho2)
%
% Author: Tamas Kis
% Last Update: 2022-05-02
%
% REFERENCES:
%   [1] Kochenderfer and Wheeler, "Algorithms for Optimization"
%       (p. 181)
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
%   rho1    - (1×1 function_handle) count penalty magnitude
%   rho2    - (1×1 function_handle) quadratic penalty magnitude
%
% -------
% OUTPUT:
% -------
%   p       - (1×1 function_handle) count penalty function, p(x) 
%             (p : ℝⁿ → ℝ)
%
%==========================================================================
function p = penalty_mixed(h,g,rho1,rho2)
    
    % defaults equality constraint to 0 if not input
    if isempty(h)
        h = @(x) 0;
    end
    
    % defaults inequality constraint to 0 if not input
    if isempty(g)
        g = @(x) 0;
    end
    
    % defines mixed penalty function
    p = @(x) rho1*penalty_count(h,g)+rho2*penalty_quadratic(h,g);
    
end