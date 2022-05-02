%==========================================================================
%
% basis  Returns the ith n-dimensional standard basis vector.
%
%   ei = basis(i,n)
%
% Author: Tamas Kis
% Last Update: 2022-05-02
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   i       - (1×1 double) specifies which standard basis vector to return
%   n       - (1×1 double) dimension of the basis vector
%
% -------
% OUTPUT:
% -------
%   ei      - (n×1 double) ith n-dimensional basis vector, eᵢ
%
%==========================================================================
function ei = basis(i,n)
    ei = zeros(n,1);
    ei(i) = 1;
end