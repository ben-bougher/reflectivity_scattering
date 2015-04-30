
function [] = extract_signals(labels, T, output_file)

addPaths;

addpath('/Users/bbougher/course_work/CPSC540/course_project')

label_dir = '/Users/bbougher/course_work/CPSC540/course_project/unit_labels/';
input_dir = '/Users/bbougher/course_work/CPSC540/course_project/grlogs/';

filter = 'gr';

las_files = dir(strcat(input_dir, '*.las'));
label_files = dir(strcat(label_dir, '*.csv'));



data = struct();
for i =1:size(labels,2)
    data.(labels{i}) = [];
end


for f = 1:size(las_files,1)

    % Check if the log has labels
    lfile =  las_files(f);
    api = lfile.name(1:10);
    
    if(sum(strcmp({label_files.name}, strcat(api, '.csv'))) == 0)
        continue;
    end
    
    wlog = read_las_file(strcat(input_dir, las_files(f).name));
    
    label_data = parsecsv(strcat(label_dir, strcat(api, '.csv')));
    labels = fieldnames(label_data);
    
    for group =1:size(labels,1)
        
        
        label = labels{group};
        range = label_data.(label);
        
        if sum(range) == 0
            continue
        end
        
        if ~(isfield(data, label))
            continue
        end
        depth = wlog.curves(:,1);
        
        
        group_index = find((depth <= range(2))  & (depth > ...
                                                   range(1)));
        
        index = find(strcmp(wlog.curve_info(:,1), filter));
        y = wlog.curves(:, index);
        y = y(group_index);
        
        if size(y,1) < T
            continue
        end
       
        n_windows = floor(size(y,1) / T);
        
        sig = reshape(y(1:n_windows * T), T, n_windows);
        data.(label) = cat(2,data.(label), sig);
        
    end
    
end

save(output_file, 'data')
