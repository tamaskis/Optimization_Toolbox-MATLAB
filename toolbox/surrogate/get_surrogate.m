%==========================================================================
%
% get_surrogate  Develops a surrogate model using radial basis functions.
%
%   f_hat = get_surrogate(f,x)
%   f_hat = get_surrogate(f,x,type)
%   f_hat = get_surrogate(f,x,type,sigma)
%
% Author: Tamas Kis
% Last Update: 2022-05-23
%
% REFERENCES:
%   [1] Kochenderfer and Wheeler, "Algorithms for Optimization"
%       (pp. 255, 259, 262-263)
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   f       - (1×1 function_handle) objective function, f(x) (f : ℝⁿ → ℝ)
%   x       - (n×m double) m design point samples
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
%   f_hat   - (1×1 function_handle) surrogate model, f_hat(x) 
%             (f_hat : ℝⁿ → ℝ)
%
%==========================================================================
function f_hat = get_surrogate(f,x,type,sigma)
    
    % defaults "type" to 'cubic'
    if (nargin < 3) || isempty(type)
        type = 'cubic';
    end
    
    % defaults "sigma" to 1
    if (nargin < 4) || isempty(sigma)
        sigma = 1;
    end
    
    % kernel for radial basis functions
    psi = kernel(type,sigma);
    
    % number of samples
    m = size(x,2);
    
    % sample points serve as centers for radial basis functions
    c = x;
    
    % preallocates design matrix
    B = zeros(m,m);
    
    % populates design matrix
    for i = 1:m
        B(i,:) = radial_basis(x(:,i),c,psi).';
    end
    
    % evaluates objective function at each design point
    y = zeros(m,1);
    for i = 1:m
        y(i) = f(x(:,i));
    end
    
    % solves for surrogate model coefficients
    theta = B\y;
    
    % surrogate model
    f_hat = @(x) (theta.')*radial_basis(x,c,psi);
    
end