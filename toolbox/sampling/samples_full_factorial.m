%==========================================================================
%
% samples_full_factorial  Full factorial sampling plan.
%
%   x = samples_full_factorial(x_min,x_max)
%
% Author: Tamas Kis
% Last Update: 2022-05-23
%
% REFERENCES:
%   [1] Kochenderfer and Wheeler, "Algorithms for Optimization"
%       (pp. 235-236)
%   [2] https://www.mathworks.com/matlabcentral/answers/263932-unknown-number-of-output-variables
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   x_min   - (n×1 double) minimum design point (i.e. minimum values of
%             each design variable)
%   x_max   - (n×1 double) maximum design point (i.e. maximum values of
%             each design variable)
%   m       - (n×1 double) number of samples in each direction
%
% -------
% OUTPUT:
% -------
%   x       - (n×M double) M samples of the design point
%
%==========================================================================
function x = samples_full_factorial(x_min,x_max,m)
    
    % design point dimension
    n = length(x_min);
    
    % 1-dimensional grids for each design variable
    x_1d = cell(1,n);
    for i = 1:n
        x_1d{i} = x_min(i):((x_max(i)-x_min(i))/m(i)):x_max(i);
    end
    
    % n-dimensional grids for each design variable
    x_nd = cell(1,n);
    [x_nd{:}] = ndgrid(x_1d{:});
    
    % defines an n×M matrix storing M samples of the n-dimensional design
    % point
    x = zeros(n,length(x_nd{i}(:)));
    for i = 1:n
        x(i,:) = x_nd{i}(:);
    end
    
end