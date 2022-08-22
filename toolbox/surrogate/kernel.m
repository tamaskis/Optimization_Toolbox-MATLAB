%==========================================================================
%
% kernel  Kernels for radial basis functions.
%
%   psi = kernel
%   psi = kernel(type)
%   psi = kernel(type,sigma)
%
% Author: Tamas Kis
% Last Update: 2022-05-23
%
% REFERENCES:
%   [1] Kochenderfer and Wheeler, "Algorithms for Optimization" (pp. 262)
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   type    - (char) (OPTIONAL) 'linear', 'cubic', 'thin plate spline', 
%             'Gaussian', 'multiquadratic', or 'inverse multiquadratic'
%             (defaults to 'cubic')
%   sigma   - (1×1 double) (OPTIONAL) hyperparameter (needed for 
%             'Gaussian', 'multiquadratic', or 'inverse multiquadratic'
%             kernels) (defaults to 1)
%
% -------
% OUTPUT:
% -------
%   psi     - (1×1 function_handle) kernel, ψ(r) (ψ : ℝ → ℝ)
%
%==========================================================================
function psi = kernel(type,sigma)
    
    % defaults "type" to 'cubic'
    if (nargin < 1) || isempty(type)
        type = 'cubic';
    end
    
    % defaults "sigma" to 1
    if (nargin < 2) || isempty(sigma)
        sigma = 1;
    end
    
    % defines kernel
    if strcmpi(type,'linear')
        psi = @(r) r;
    elseif strcmpi(type,'cubic')
        psi = @(r) r^3;
    elseif strcmpi(type,'thin plate spline')
        psi = @(r) r^2*log(r);
    elseif strcmpi(type,'Gaussian')
        psi = @(r) exp(-r^2/(2*sigma^2));
    elseif strcmpi(type,'multiquadratic')
        psi = @(r) sqrt(r^2+sigma^2);
    elseif strcmpi(type,'inverse multiquadratic')
        psi = @(r) (r^2+sigma^2)^(-1/2);
    end
    
end