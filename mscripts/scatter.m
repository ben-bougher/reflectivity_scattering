
function S = scatter(data, N, T, M)

    scat_coeffs = [];
    filt_opt = default_filter_options('dyadic', T);

    % Only compute zeroth-, first- and second-order scattering.
    scat_opt.M = 2;
    
    % Prepare wavelet transforms to use in scattering.
    [Wop, filters] = wavelet_factory_1d(N, filt_opt, scat_opt);

    try
        % Compute the scattering coefficients of y.
        S = scat(data, Wop);
   
            
    catch err
        err
    end
    
end 
    
    







