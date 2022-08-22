%==========================================================================
%
% radial_basis  Evaluates m radial basis functions.
%
%   b = radial_basis(x,c,psi)
%
% Author: Tamas Kis
% Last Update: 2022-05-23
%
% REFERENCES:
%   [1] Kochenderfer and Wheeler, "Algorithms for Optimization"
%       (pp. 255, 259, 262-263)
%   [2] https://en.wikipedia.org/wiki/Radial_basis_function
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   x       - (n×1 double) design point
%   c       - (n×m double) centers of radial basis function
%   psi     - (1×1 function_handle) kernel
%
% -------
% OUTPUT:
% -------
%   b       - (m×1 double) evaluation of radial basis functions at x
%
%==========================================================================
function b = radial_basis(x,c,psi)
    
    % number of radial basis functions
    m = size(c,2);
    
    % preallocates vector b to store evaluations of radial basis functions
    b = zeros(m,1);
    
    % evaluates radial basis functions
    for i = 1:m
        b(i) = psi(inorm(x-c(:,i)));
    end
    
end