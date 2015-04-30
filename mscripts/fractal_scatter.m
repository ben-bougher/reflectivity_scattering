function [] = fractal_scatter(infile, outfile, N, M, T)

load(infile);
addPaths;

X = [];
y=[];
data
depth = max(M);
labels = fieldnames(data);
for i =1:size(labels,1)
    
    label = labels{i};
    signals = data.(label);
    
    for j= 1:size(signals,2)
        S = scatter(signals(:,j), N, T, depth);
 
        x = [];
        for k=1:size(M,2)
            
            x = [x,S{M(k) + 1}.signal{:}];
        end
        
        y = [y,labels(i)];   
        X = [X; x];
    end

end

save(outfile, 'X', 'y')



    
    
