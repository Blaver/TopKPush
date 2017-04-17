%Usage: 
%load 'xxx_result.mat' manually, then this script could use 
%the raw 'np_result' array to calculate 
%average np-score and corresponding std for each method.

mydata = auc_result;
[nt, nr] = size(mydata);
meanss = [];
stdss = []; 
for t = 1:nt
    n_mhd = size(mydata{1, 1}, 2);
    
    means = zeros(1, n_mhd);
    stds = zeros(1, n_mhd);
    ps = zeros(n_mhd, n_mhd);
    
    data = [];
    for r = 1:nr
        data = [data; mydata{t, r}];
    end
    
    %for np_result, we may want to cut off the value when > 1
    %data = min(data, 1);
    
    means = mean(data, 1);
    stds = std(data, 0, 1);
    meanss  = [meanss; means];
    stdss = [stdss; stds];
end