load '../data/reflectivity.mat'
addPaths;


seis = seis_time';
ref = ref_time';

% Length of data to use
N = size(seis_time,2);

% Scattering window size
T = 1024;

% depth of scattering network
M = 2;

scat_seis = scatter(seis,N,T,M);
scat_ref = scatter(ref,N,T,M);


    

t = [0:nsamps-1] * dt;
figure

plot(t, ref);%title('reflectivity series', 'FontSize', 20);
xlabel('time [s]','FontSize', 16);
ylabel('amplitude');xlim([0, max(t)])
ylim([-8,8]);

figure

plot(t,seis);%title('synthetic seismic', 'FontSize', 20)
;xlabel('time [s]','FontSize', 16);
ylabel('amplitude');xlim([0, max(t)]);ylim([-.5,.5]);

l1 = strtrim(cellstr(num2str(scat_seis{M+1}.meta.j(1,:)')))
l2 = strtrim(cellstr(num2str(scat_seis{M+1}.meta.j(2,:)')))
l = l1 %strcat(l1,'|',l2);

t = linspace(0, max(t), size([scat_ref{M+1}.signal{:}]',2));
c = [0:size([scat_ref{M+1}.signal{:}],2)-1]

figure
imagesc(t,c, [scat_ref{M+1}.signal{:}]');xlabel('time [s]', 'FontSize', 16);
%title('reflectivity scattering', 'FontSize', 20);
set(gca, 'YTick', []);xlim([0, max(t)])

figure
imagesc(t,c,[scat_seis{M+1}.signal{:}]');xlabel('time [s]', 'FontSize', 16);
%title('seismic scattering', 'FontSize', 20);
set(gca, 'YTick', []);xlim([0, max(t)])



    


    
    
    


    
    