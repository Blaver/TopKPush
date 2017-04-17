function [svm_model] = BS_SVM_ThresholdSelect( ytr, Xtr,  opt )
%BS_SVM_THRESHOLDSELECT 此处显示有关此函数的摘要
%   此处显示详细说明
%% Option parsing and parameter initialization
    if ~isfield(opt,'target') opt.target = 'NP_SCORE'; end
    if ~isfield(opt,'t') opt.t = 0.1; end
    if ~isfield(opt,'is_fulltrain') opt.is_fulltrain = 1; end
    if ~isfield(opt,'valid_rate') opt.valid_rate = 1/3; end
    if ~isfield(opt,'c') opt.c = 1; end
    if ~isfield(opt,'e') opt.e = 0.001; end
    if ~isfield(opt,'liblinear') opt.liblinear = 0; end
    
    target_name = opt.target;
    seed = opt.seed;
    t = opt.t;
    is_fulltrain = opt.is_fulltrain;
    valid_rate = opt.valid_rate;
    c = opt.c;
    e = opt.e;
    liblinear = opt.liblinear;
    
    num = size(ytr, 1);
    rng(seed);
    order = randperm(num, num);
    rng('shuffle');
    X = Xtr(order, :);
    Y = ytr(order, :);
%% Select threshold
    if ~is_fulltrain     
        valid_indices = 1:ceil(num*valid_rate);
        train_indices = (1 + ceil(num*valid_rate)):num;
      
        %training data
        X_train = X(train_indices, :);
        Y_train = Y(train_indices, :);
        %validation data
        X_valid = X(valid_indices, :);
        Y_valid = Y(valid_indices, :);
    else
        X_train = X;
        Y_train = Y;
        X_valid = X;
        Y_valid = Y;
    end
    
    if ~liblinear
        %train a single svm model
        svm_model = libsvmtrain(Y_train, X_train, ['-s 0 -t 0 -c ', num2str(c), ' -e ', num2str(e), ' -q']);
        %scoring and predicting on the valid set 
        [~, ~, decision_values] = libsvmpredict(Y_valid, X_valid, svm_model);
    else
        %train a single svm model
        svm_model = liblineartrain(Y_train, X_train, ['-s 3 -c ', num2str(c), ' -e ', num2str(e), ' -B 1 -q']);
        %scoring and predicting on the valid set 
        [~, ~, decision_values] = liblinearpredict(Y_valid, X_valid, svm_model, '-q');
    end
    
    %for BS-SVM with np-score
    if strcmp(target_name, 'NP_SCORE')
        [rank_list, rank_i] = sort(decision_values);
        rank_label = Y_valid(rank_i);

        M = size(Y_valid, 1);
        neg_n = sum(Y_valid == -1);
        pos_n = sum(Y_valid == 1);

        %in the beginning, rho is a bit smaller than the left
        %endpoint of the ranked score list, this time np-score is 
        %(1 - t)/t.
        %Be careful that here we do not consider the situation of np_score
        %== 1 which means that ALL the samples will be treated as negative samples since it is useless.
        best_thd = rank_list(1) - 1e-3;
        best_np = (1 - t)/t;
        FP = neg_n;
        TP = pos_n;
        for i = 1:M
            if i == 1 || (i > 1 && rank_list(i) ~= rank_list(i -1))
                fpr = FP/neg_n;
                tpr = TP/pos_n;
                np = max(fpr/t, 1) - tpr;
                if np < best_np
                    best_np = np;
                    best_thd = 0.5*(rank_list(i) + rank_list(max(1, i - 1)));   
                end
            end
            
            if (rank_label(i)  == 1)
                TP = TP - 1;
            else 
                FP = FP - 1;
            end
        end
        
        %update bias, for libsvm, 
        if ~liblinear
            svm_model.rho = svm_model.rho + best_thd;
        else
            svm_model.w(end) = svm_model.w(end) - best_thd;
        end
    end
        
    %for BS-SVM with best tpr
    if strcmp(target_name, 'BTPR')
        [rank_list, rank_i] = sort(decision_values);
        rank_label = Y_valid(rank_i);

        M = size(Y_valid, 1);
        neg_n = sum(Y_valid == -1);
        pos_n = sum(Y_valid == 1);

        %in the beginning, rho is a bit larger than the right
        %endpoint of the ranked score list, this time tpr is 0.
        %Be careful that here we do not consider the situation of np_score
        %== 1 which means that ALL the samples will be treated as negative samples since it is useless.
        best_thd = rank_list(end) + 1e-3;
        best_tpr = 0;
        FP = 0;
        TP = 0;
        for i = M:-1:1
            if (rank_label(i)  == 1)
                TP = TP + 1;
            else 
                FP = FP + 1;
            end
            
            fpr = FP/neg_n;
            tpr = TP /pos_n;
%             fpr = sum(rank_label(i:M) == -1)/neg_n;
%             tpr = sum(rank_label(i:M) == 1)/pos_n;
            if fpr > t break; end
            if tpr > best_tpr
                best_tpr = tpr;
                best_thd = 0.5*(rank_list(i) + rank_list(max(1, i - 1)));   
            end        
        end

        %update bias, for libsvm
        if ~liblinear
            svm_model.rho = svm_model.rho + best_thd;
        else
            svm_model.w(end) = svm_model.w(end) - best_thd;
        end
    end
end

