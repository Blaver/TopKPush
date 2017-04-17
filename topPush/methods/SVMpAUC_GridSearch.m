function [ best_lambda, best_target ] = SVMpAUC_GridSearch( Xtr, ytr, opt )
%GRID SEARCH
    %% Option parsing and parameter initialization
    if ~isfield(opt,'n_fold')  opt.n_fold = 2; end
    if ~isfield(opt,'target') opt.target = 'BTPR'; end
    if ~isfield(opt,'t') opt.t = 1; end
    %this option is only used when n_fold == 1
    if ~isfield(opt,'valid_rate') opt.valid_rate = 0.3; end
    
    n_fold = opt.n_fold;
    seed = opt.seed;
    target_name = opt.target;
    t = opt.t;
    scale = opt.scale;
    valid_rate = opt.valid_rate;
    
%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ~isfield(opt,'lambda_lb') opt.lambda_lb = 1e-3; end
    if ~isfield(opt,'lambda_ub') opt.lambda_ub = 1e3; end
    if ~isfield(opt,'lambda_factor') opt.lambda_factor = 10; end
    
    lambda_lb = opt.lambda_lb;
    lambda_ub = opt.lambda_ub;
    lambda_factor = opt.lambda_factor;
%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Grid Search
    num = size(ytr, 1);
    X = Xtr;
    Y = ytr;
%     rng(seed);
%     order = randperm(num, num);
%     rng('shuffle');
%     X = Xtr(order, :);
%     Y = ytr(order, :);
   
%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    best_target = -inf;
    best_lambda = -inf;
    
    %these bounds are dynamic.
    dyn_lambda_lb = lambda_lb;
    dyn_lambda_ub = lambda_ub;
    %search direction of lambda
    lambda_sd = 0;
    
    lambda = lambda_lb;
    while  lambda <= dyn_lambda_ub
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
             if strcmp(target_name, 'T_AUC')
                ttt = 0;
                tt = max(t, 1/sum(Y_train == -1) * scale);
            elseif strcmp(target_name, 'BTPR')
                ttt = max(0, t -1/sum(Y_train == -1) * scale);
                tt = max(t, 1/sum(Y_train == -1) * scale);
             end
            w = SVMpAUC_TIGHT(X_train, Y_train, ttt , tt, lambda, 1e-4, 0);
            decision_values = X_valid*w;
            
%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if strcmp(target_name, 'T_AUC')
                t_auc = calculate_roc(decision_values, Y_valid, 1, -1, t);
                sum_target = sum_target + t_auc;
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
        end
        
        %% Bounday Extend Related Codes
        %case 1: 
        %   initial search of lambda is complete, but boundary extending 
        %   is not applied yet.  Check whether we need extend boundary of lambda.
        if lambda_sd == 0  && lambda == lambda_ub
            %extend left bound
            if best_lambda == dyn_lambda_lb
                dyn_lambda_lb = dyn_lambda_lb/lambda_factor;
                dyn_lambda_ub = dyn_lambda_lb;
                lambda_sd = -1;
            elseif best_lambda == dyn_lambda_ub
                dyn_lambda_ub = dyn_lambda_ub*lambda_factor;
                dyn_lambda_lb = dyn_lambda_ub;
                lambda_sd = 1;
            end
        end
        %case 2:
        %   searching outside the left boundary of w.
        if lambda_sd == -1 && best_lambda == dyn_lambda_lb
            dyn_lambda_lb = dyn_lambda_lb/lambda_factor;
            dyn_lambda_ub = dyn_lambda_lb;
        end
        %case 3:
        %   searching outside the right boundary  of c.
        if lambda_sd == 1 && best_lambda == dyn_lambda_ub
            dyn_lambda_ub = dyn_lambda_ub*lambda_factor;
            dyn_lambda_lb = dyn_lambda_ub;
        end
        %%
        %update lambda
        lambda = lambda * lambda_factor;
%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end  
    best_target = abs(best_target)/n_fold;

end

