load '../data/reflectivity.mat'
addPaths;


time = seis_time';
depth = seis_depth';

% Length of data to use
N = size(time,1);

% Scattering window size
T = 128;

% depth of scattering network
M = 2;

time_s = scatter(time,N,T,M);
depth_s = scatter(depth,N,T,M);


for i=1:M+1
    
    figure
    time_coeffs = [time_s{i}.signal{:}]';
    depth_coeffs = [depth_s{i}.signal{:}]';
    
    subplot(221);
    plot(time);
        
    subplot(222);

    plot(depth);

    if i > 1
        subplot(223);
        imagesc(time_coeffs);
    
        subplot(224);
        imagesc(depth_coeffs);
    else
        subplot(223);
        plot(time_coeffs);
    
        subplot(224);
        plot(depth_coeffs);
    end
    
end

    
    