function [ OOBModel ] = OOB_FindThreshold( OOBModel, TH_method, t)
    bn = size(OOBModel.w, 2);
    %cache that restore learned thresholds of each bagger
    B = zeros(1, bn);
    
    for i = 1:bn
        rk_list = OOBModel.S(:, i);
        rk_list_labels = OOBModel.L(:, i);
        
        if strcmp(TH_method, 'hard')
            thd = find_hard_threshold(rk_list, rk_list_labels, t);
        elseif strcmp(TH_method, 'soft')
            thd = find_soft_threshold(rk_list, rk_list_labels, t);
        end
        B(i) = thd;
    end
    OOBModel.b = B;
end

%% Hard threshold finding method, follow the user tolerance rule strictly.
function thd = find_hard_threshold(rk_list, rk_list_labels, t)
    %reverse rk_list for convenience
    rk_list = flip(rk_list);
    rk_list_labels = flip(rk_list_labels);
    
    n = size(rk_list_labels, 1);
    %false positive tolerance upper bound 
    UB = floor(sum(rk_list_labels == -1) * t);

    %at the beginning, thd is a few larger than all other scores 
    % postive: >= thd;
    % negtive: < thd.
    % so, FP is zero in the beginning. 
    thd = rk_list(1) + 1e-4;
    FP = 0;
    for j = 1:n
        %first, update FP 
        if rk_list_labels(j) == -1
            FP = FP + 1;
        end
        %then check if FP is overflowed. if so, stop the line search process and accept current thd.
        if FP > UB
            break;
        end
        %update thd
        if j == n || rk_list(j) ~= rk_list(j + 1)
            thd = rk_list(j);
        end
    end
end

%% Soft threshold finding method, find threshold that minimize Neyman-Pearson Score
function [thd, min_np_score] = find_soft_threshold(rk_list, rk_list_labels, t)    
    %% method2 only achieves O(n) time complexity.
    %reverse rk_list for convenience
    rk_list = flip(rk_list);
    rk_list_labels = flip(rk_list_labels);
    
    n_data = size(rk_list_labels, 1);
    nN = sum(rk_list_labels == -1);
    nP = sum(rk_list_labels == 1);
    
    %At the beginning, thd is a few larger than all other scores 
    % postive: >= thd;
    % negtive: < thd.
    %so, at this time np_score is max{fpr/t, 1} - tpr = 1
    thd = rk_list(1) + 1e-4;
    min_np_score = 1;
    
    FP = 0;
    for i = 1:n_data
        if rk_list_labels(i) == -1
            FP = FP + 1;
        end
        % if we arrive the ending of a new threshold, start comparing.
        if i == n_data || rk_list(i) ~= rk_list(i + 1) 
            FPR = FP/nN;
            TPR = (i - FP)/nP;
            np_score = max(FPR/t, 1) - TPR;
            if np_score < min_np_score
                min_np_score = np_score;
                thd = rk_list(i);
            end    
        end
    end
end

