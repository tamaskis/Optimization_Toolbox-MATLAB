%==========================================================================
%
% minimize_nelder_mead  Finds the local minimizer and minimum of an 
% objective function using the Nelder-Mead simplex method.
%
%   x_min = minimize_hooke_jeeves(f,x0,sigma0)
%   x_min = minimize_hooke_jeeves(f,x0,sigma0,opts)
%   [x_min,f_min] = minimize_hooke_jeeves(__)
%   [x_min,f_min,k] = minimize_hooke_jeeves(__)
%   [x_min,f_min,k,x_all,f_all] = minimize_hooke_jeeves(__)
%
% Author: Tamas Kis
% Last Update: 2022-05-02
%
% REFERENCES:
%   [1] Kochenderfer and Wheeler, "Algorithms for Optimization"
%       (pp. 105-108, 110)
%
%--------------------------------------------------------------------------
%
% ------
% INPUT:
% ------
%   f       - (1×1 function_handle) objective function, f(x) (f : ℝⁿ → ℝ)
%   x0      - (n×1 double) initial guess for local minimizer
%   opts    - (1×1 struct) (OPTIONAL) solver options
%       • alpha      - (1×1 double) reflection parameter (defaults to 1)
%       • beta       - (1×1 double) expansion parameter (defaults to 2)
%       • gamma      - (1×1 double) contraction parameter (defaults to 0.5)
%       • k_max      - (1×1 double) maximimum number of iterations
%                      (defaults to 200)
%       • sigma0     - (1×1 double) standard deviation for creating 
%                      initial simplex through random sampling (defaults to
%                      10)
%       • return_all - (1×1 logical) all intermediate root estimates are
%                      returned if set to "true"; otherwise, a faster 
%                      algorithm is used to return only the converged local
%                      minimizer/minimum (defaults to false)
%       • TOL        - (1×1 double) tolerance (defaults to 10⁻¹⁰)
%
% -------
% OUTPUT:
% -------
%   x_min   - (n×1 double) local minimizer of f(x)
%   f_min   - (1×1 double) local minimum of f(x)
%   k       - (1×1 double) number of solver iterations
%   x_all   - (n×k double) all estimates of local minimizer of f(x)
%   f_all   - (1×k double) all estimates of local minimum of f(x)
%
% -----
% NOTE:
% -----
%   --> α > 0 
%   --> β > max(1,α)
%   --> γ ∈ (0,1)
%
%==========================================================================
function [x_min,f_min,k,x_all,f_all] = minimize_nelder_mead(f,x0,opts)
    
    % ----------------------------------
    % Sets (or defaults) solver options.
    % ----------------------------------
    
    % sets reflection parameter (defaults to 1)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'alpha')
        alpha = 1;
    else
        alpha = opts.alpha;
    end
    
    % sets expansion parameter (defaults to 2)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'beta')
        beta = 1;
    else
        beta = opts.beta;
    end
    
    % sets contraction parameter (defaults to 0.5)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'gamma')
        gamma = 0.5;
    else
        gamma = opts.gamma;
    end
    
    % sets maximum number of iterations (defaults to 200)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'k_max')
        k_max = 200;
    else
        k_max = opts.k_max;
    end
    
    % sets initial simplex standard deviation (defaults to 10)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'sigma0')
        sigma0 = 10;
    else
        sigma0 = opts.sigma0;
    end
    
    % determines if all intermediate estimates should be returned
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'return_all')
        return_all = false;
    else
        return_all = opts.return_all;
    end
    
    % sets tolerance (defaults to 10⁻¹⁰)
    if (nargin < 3) || isempty(opts) || ~isfield(opts,'TOL')
        TOL = 1e-10;
    else
        TOL = opts.TOL;
    end
    
    % ---------------------
    % Optimization routine.
    % ---------------------
    
    % preallocates arrays
    if return_all
        x_all = zeros(length(x0),k_max+1);
        f_all = zeros(1,k_max+1);
    end
    
    % design point dimension
    n = length(x0);
    
    % randomly samples initial simplex
    S = randmvn(x0,diag(sigma0^2*ones(n,1)),n+1);

    % evaluates objective function at initial simplex
    y_arr = zeros(n,1);
    for i = 1:(n+1)
        y_arr(i) = f(S(:,i));
    end
    
    % Nelder-Mead simplex method
    for k = 1:k_max
        
        % stores results in arrays
        if return_all
            [f_all(k),i_min] = min(y_arr);
            x_all(:,k) = S(:,i_min);
        end
        
        [y_arr,p] = sort(y_arr); % sort lowest to highest
        S = S(:,p);
        xl = S(:,1); yl = y_arr(1); % lowest point
        xh = S(:,end); yh = y_arr(end); % highest point
        ys = y_arr(end-1); % second-highest point
        xm = sample_mean(S(1:(end-1))); % centroid
        xr = xm+alpha*(xm-xh); yr = f(xr); % reflection point
        if yr < yl
            xe = xm+beta*(xr-xm); % expansion point
            ye = f(xe);
            if ye < yr
                S(:,end) = xe;
                y_arr(end) = ye;
            else
                S(:,end) = xr;
                y_arr(end) = yr;
            end
        elseif yr >= ys
            if yr < yh
                xh = xr;
                yh = yr;
                S(:,end) = xr;
                y_arr(end) = yr;
            end
            xc = xm+gamma*(xh-xm); % contraction point
            yc = f(xc);
            if yc > yh
                for i = 2:length(y_arr)
                    S(:,i) = (S(:,i)+xl)/2;
                    y_arr(i) = f(S(:,i));
                end
            else
                S(:,end) = xc;
                y_arr(end) = yc;
            end
        else
            S(:,end) = xr;
            y_arr(end) = yr;
        end
        
        % terminates if standard deviation of y_arr falls below tolerance
        if std(y_arr) < TOL
            break;
        end
        
    end
    
    % converged local minimizer and minimum
    [f_min,i_min] = min(y_arr);
    x_min = S(:,i_min);
    
    % stores converged results and trims arrays
    if return_all
        x_all(:,k+1) = x_min; x_all = x_all(:,1:(k+1));
        f_all(k+1) = f_min; f_all = f_all(:,1:(k+1));
    end
    
end