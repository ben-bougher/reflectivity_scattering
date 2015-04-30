function [] = scattering_transfer(infile, outfile,N,T)

load(infile);
addPaths;

X = [];
y=[];
data
depth = 2;
labels = fieldnames(data);
for i =1:size(labels,1)
    
    label = labels{i};
    signals = data.(label);
    
    for j= 1:size(signals,2)
        
        S = scatter(signals(:,j), N, T, depth);

        Sj1j2 = [S{3}.signal{:}];
        
        Sj1 = [S{2}.signal{:}];
        % Scattering transfer(Sj1j2/Sj1)
        scales = S{3}.meta.j;
        
        x = Sj1j2 ./ Sj1(scales(1,:) +1);
        
        y = [y,labels(i)];   
        X = [X; x];
    end

end

save(outfile, 'X', 'y')