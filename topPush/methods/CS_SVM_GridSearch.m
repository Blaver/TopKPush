function [ best_c, best_w, best_target ] = CS_SVM_GridSearch(ytr, Xtr,  opt)
%GRID SEARCH
    %% Option parsing and parameter initialization
    if ~isfield(opt,'n_fold')  opt.n_fold = 2; end
    if ~isfield(opt,'target') opt.target = 'NP_SCORE'; end
    if ~isfield(opt,'t') opt.t = 0.1; end
    %this option is only used when n_fold == 1
    if ~isfield(opt,'valid_rate') opt.valid_rate = 0.3; end
    if ~isfield(opt,'e') opt.e = 0.001; end
    if ~isfield(opt,'liblinear') opt.liblinear = 0; end
    
    n_fold = opt.n_fold;
    seed = opt.seed;
    target_name = opt.target;
    t = opt.t;
    valid_rate = opt.valid_rate;
    e = opt.e;
    liblinear = opt.liblinear;
    
%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %base cost
    if ~isfield(opt,'c_lb') opt.c_lb = 1e-3; end
    if ~isfield(opt,'c_ub') opt.c_ub = 1e3; end
    if ~isfield(opt,'c_factor') opt.c_factor = 3; end
    %w = C+/C-, w <= 1 is suitable for low fp. 
    if ~isfield(opt,'w_lb') opt.w_lb = 1e-3; end
    if ~isfield(opt,'w_ub') opt.w_ub = 1; end
    if ~isfield(opt,'w_factor') opt.w_factor = 3; end
    
    c_lb =  opt.c_lb;
    c_ub = opt.c_ub;
    c_factor = opt.c_factor;
    w_lb =  opt.w_lb;
    w_ub = opt.w_ub;
    w_factor = opt.w_factor;
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
    
    best_target = -inf;
    
%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    best_c = -1;
    best_w = -1;
     
    %these  bounds are dynamic.
    dyn_c_lb = c_lb;
    dyn_c_ub = c_ub;
    dyn_w_lb = w_lb;
    dyn_w_ub = w_ub;
    %search direction of c and w
    c_sd = 0;
    w_sd = 0;
   
    c = c_lb;
    while c <= dyn_c_ub
        w = w_lb;
        while w <= dyn_w_ub
%%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if ~liblinear 
                    %train a single svm model
                    svm_model = libsvmtrain(Y_train, X_train, ['-s 0 -t 0 -c ', num2str(c), ' -w1 ', num2str(w), ' -e ', num2str(e),  ' -q']);
                    %predict and calculate target
                    [pdt_l, accuracy, decision_values] = libsvmpredict(Y_valid, X_valid, svm_model);
                else
                    %train a single svm model
                    svm_model = liblineartrain(Y_train, X_train, ['-s 3 -c ', num2str(c), ' -w1 ', num2str(w), ' -e ', num2str(e),  ' -q']);
                    %predict and calculate target
                    [pdt_l, accuracy, decision_values] = liblinearpredict(Y_valid, X_valid, svm_model, '-q');
                end
%%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
                best_c = c;
                best_w = w;
            end
            
            %% Bounday Extend Related Codes
            %case 1: 
            %   initial search of w is complete, but boundary extending 
            %   is not applied yet.  Check whether we need extend boundary of w.
            if w_sd == 0  && w == w_ub
                %extend left bound
                if best_w == dyn_w_lb
                    dyn_w_lb = dyn_w_lb/w_factor;
                    dyn_w_ub = dyn_w_lb;
                    w_sd = -1;
                elseif best_w == dyn_w_ub
                    dyn_w_ub = dyn_w_ub*w_factor;
                    dyn_w_lb = dyn_w_ub;
                    w_sd = 1;
                end
            end
            %case 2:
            %   searching outside the left boundary of w.
            if w_sd == -1 && best_w == dyn_w_lb
                dyn_w_lb = dyn_w_lb/w_factor;
                dyn_w_ub = dyn_w_lb;
            end
            %case 3:
            %   searching outside the right boundary  of w.
            if w_sd == 1 && best_w == dyn_w_ub
                dyn_w_ub = dyn_w_ub*w_factor;
                dyn_w_lb = dyn_w_ub;
            end        
            w = w*w_factor;
        end 
        
        %% Bounday Extend Related Codes
        %case 1: 
        %   initial search of c is complete, but boundary extending 
        %   is not applied yet.  Check whether we need extend boundary of c.
        if c_sd == 0  && c == c_ub
            %extend left bound
            if best_c == dyn_c_lb
                dyn_c_lb = dyn_c_lb/c_factor;
                dyn_c_ub = dyn_c_lb;
                c_sd = -1;
            elseif best_c == dyn_c_ub
                dyn_c_ub = dyn_c_ub*c_factor;
                dyn_c_lb = dyn_c_ub;
                c_sd = 1;
            end
        end
        %case 2:
        %   searching outside the left boundary of w.
        if c_sd == -1 && best_c == dyn_c_lb
            dyn_c_lb = dyn_c_lb/c_factor;
            dyn_c_ub = dyn_c_lb;
        end
        %case 3:
        %   searching outside the right boundary  of c.
        if c_sd == 1 && best_c == dyn_c_ub
            dyn_c_ub = dyn_c_ub*c_factor;
            dyn_c_lb = dyn_c_ub;
        end
        %%
        c = c*c_factor;
%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end  
    best_target = abs(best_target)/n_fold;
end

