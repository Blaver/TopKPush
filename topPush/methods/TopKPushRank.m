function [best_lambda, best_K, best_target]= topKPushRank(Xtr, ytr,  opt)
%GRID SEARCH
    %% Option parsing and parameter initialization
    if ~isfield(opt,'n_fold')  opt.n_fold = 2; end
    if ~isfield(opt,'target') opt.target = 'BTPR'; end
    if ~isfield(opt,'t') opt.t = 0.1; end
    %this option is only used when n_fold == 1
    if ~isfield(opt,'valid_rate') opt.valid_rate = 0.3; end
    
    n_fold = opt.n_fold;
    target_name = opt.target;
    t = opt.t;
    valid_rate = opt.valid_rate;
    
%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ~isfield(opt,'K_on') opt.K_on = 1; end
    if ~isfield(opt,'lambda_lb') opt.lambda_lb = 1e-3; end
    if ~isfield(opt,'lambda_ub') opt.lambda_ub = 1e3; end
    if ~isfield(opt,'lambda_factor') opt.lambda_factor = 10; end
    if ~isfield(opt,'k') opt.k = 1; end
    if ~isfield(opt,'search_k') opt.search_k = 0; end
    
    lambda_lb = opt.lambda_lb;
    lambda_ub = opt.lambda_ub;
    lambda_factor = opt.lambda_factor;
    
    k_on = opt.K_on;
    
    if opt.search_k
        K_lb = 1e-4;
        K_ub = 0.1;
        K_factor  = 10;
    else
        K_lb = opt.k;
        K_ub = opt.k;
        K_factor = 10;
    end

%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Grid Search
    num = size(ytr, 1);
    order = randperm(num, num);
    X = Xtr(order, :);
    Y = ytr(order, :);
    
    best_target = -inf;
%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    best_target = -inf;
    best_lambda = -inf;
    best_K = -inf;
    
    lambda = lambda_lb;
    K = K_lb;
    
    while  lambda <= lambda_ub
        while K <= K_ub
%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        sum_target = 0; 
        for i = 1:n_fold
            %if 1_fold, split X into train-set and valid-set by valid_rate. 
            if n_fold == 1
                valid_indices = i:ceil(num*valid_rate);
                train_indices = (1 + ceil(num*valid_rate)):num;
            else
                valid_indices = i:n_fold:num;
                train_indices = setdiff(1:num, valid_indices);
            end
            %training data
            X_train = X(train_indices, :);
            Y_train = Y(train_indices, :);
            %validation data
            X_valid = X(valid_indices, :);
            Y_valid = Y(valid_indices, :);

%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%             
            wkr_opt.lambda =  lambda;
            wkr_opt.maxIter = 10000;
            wkr_opt.tol = 1e-4;
            wkr_opt.debug = false;
            wkr_opt.k = K;
            
            %training
            if k_on
                topk_model = topKPushPreciseTrain(X_train, Y_train, wkr_opt);
            else
                topk_model = topPushTrain(X_train, Y_train, wkr_opt);
            end
            %validation
            decision_values = modelPredict(topk_model, X_valid, 'SCORE');
%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if strcmp(target_name, 'ACCURACY')
                sum_target = sum_target + accuracy;
            elseif strcmp(target_name, 'NP_SCORE')
                %be careful here sum_target is negative, because we need
                %minimize np-score, and return ABS when return.
                sum_target = sum_target - neyman_pearson_score(pdt_l, Y_valid, t);
            elseif strcmp(target_name, 'T_AUC')
                t_auc = calculate_roc(decision_values, Y_valid, 1, -1, t);
                sum_target = sum_target + t_auc;
            elseif strcmp(target_name, 'AUC')
                auc = calculate_roc(decision_values, Y_valid, 1, -1, 1.0);
                sum_target = sum_target + auc;
            elseif strcmp(target_name, 'BTPR')
                [~, ~, ~, btpr] = calculate_roc(decision_values, Y_valid, 1, -1, t);    
                sum_target = sum_target + btpr;
            end
        end
        %Is this parameter setting is better?  
        if best_target < sum_target && sum_target ~= -1.0*n_fold
            best_target = sum_target;
%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            best_lambda = lambda;
            best_K = K;
        end
        K = K * K_factor;
        end
        %update lambda
        lambda = lambda * lambda_factor;
%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end  
    best_target = abs(best_target)/n_fold;
end



