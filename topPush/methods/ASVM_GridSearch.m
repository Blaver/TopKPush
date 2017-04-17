function [ best_mu, best_tau, best_target ] = ASVM_GridSearch( ytr, Xtr,  opt)
%GRIDSEARCH 
    %% Option parsing and parameter initialization
    if ~isfield(opt,'n_fold')  opt.n_fold = 2; end
    if ~isfield(opt,'target') opt.target = 'NP_SCORE'; end
    if ~isfield(opt,'t') opt.t = 0.1; end
    %this option is only used when n_fold == 1
    if ~isfield(opt,'valid_rate') opt.valid_rate = 0.3; end
    if ~isfield(opt,'e') opt.e = 0.001; end
    
    n_fold = opt.n_fold;
    seed = opt.seed;
    target_name = opt.target;
    t = opt.t;
    valid_rate = opt.valid_rate;
    e = opt.e;
    
%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ~isfield(opt,'init_mu') opt.init_mu = 1e-3; end
    if ~isfield(opt,'mu_factor') opt.mu_factor = 3; end
    if ~isfield(opt,'init_tau') opt.init_tau = 1e-3; end
    if ~isfield(opt,'tau_factor') opt.tau_factor = 3; end
    
    init_mu =  opt.init_mu;
    mu_factor = opt.mu_factor;
    init_tau = opt.init_tau;
    tau_factor = opt.tau_factor;
%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    num = size(ytr, 1);
    rng(seed);
    order = randperm(num, num);
    rng('shuffle');
    X = Xtr(order, :);
    Y = ytr(order, :);
    
    best_target = -inf;
    
%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    neg_num = sum(ytr == -1);
    pos_num = sum(ytr == 1);
    %mu + tau upper bound
    sum_ub = pos_num/num;
    %mu upper bound
    mu_ub = neg_num/num;
    
    best_mu = -1;
    best_tau = -1;
    
    %these  bounds are dynamic.
    dyn_mu_lb = init_mu;
    dyn_mu_ub = mu_ub;
    dyn_tau_lb = init_tau;
    dyn_tau_ub = init_tau;
    %search direction of c and w
    mu_sd = 0;
    tau_sd = 0;
       
    mu = init_mu;
    while mu < mu_ub
        tau = init_tau;
        while tau < (sum_ub - mu)
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
                %train a single asvm model
                asvm_model = libsvmtrain(Y_train, X_train, ['-s 5 -t 0 -x ', num2str(mu), ' -y ', num2str(tau), ' -e ', num2str(e), ' -q']);
                %this solution is infeasible, abandon it.
                if abs(asvm_model.rho + 123456.7) < 1e-3
                    sum_target = -inf;
                    break;
                end
                %predict and calculate target
                [pdt_l, accuracy, decision_values] = libsvmpredict(Y_valid, X_valid, asvm_model);
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
                best_mu = mu;
                best_tau = tau;
            end
            tau = tau*tau_factor;
        end
        tau = init_tau;
        mu = mu*mu_factor;
%%%%% CLASS BASED PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end  
    best_target = abs(best_target)/n_fold;
end

