    clc 
    clear 
    addpath(genpath(pwd))
    
datasets = {
        {'data/australian.txt', '', 1, -1, 'australian'}, ...
        {'data/breast-cancer.txt', '', 2, 4, 'breast_cancer'}, ...
        {'data/colon-cancer', '', 1, -1, 'colon-cancer'}, ...
        {'data/diabetes.txt', '', 1, -1, 'diabetes'}, ...
        {'data/german.numer.txt', '', 1, -1, 'german'}, ...
        {'data/heart.txt', '', 1, -1, 'heart'}, ...
        {'data/ionosphere.txt', '', 1, -1, 'ionosphere'}, ...
        {'data/leu', 'data/leu.t', 1, -1, 'leu'}, ...
        {'data/liver-disorders.txt', 'data/liver-disorders.t', 1, 0, 'liver-disorders'}, ...
        {'data/a8a.txt', 'data/a8a.t', 1, -1, 'a8a'}, ... 
        {'data/news20.binary', '', 1, -1, 'news20'}, ...
    };

%datasets = {
%   {'data/german.numer.txt', '', 1, -1, 'german'}, ...
%};
%% Read data into memory
    %For data file
for  data_cell = datasets  
    auc_result = cell(0, 0);
    btpr_result = cell(0, 0);
    np_result = cell(0, 0);
    fpr_result = cell(0, 0);
    tpr_result = cell(0, 0);
    
    for round = 1:10
        data_meta = data_cell{1};
        opt.train_file = data_meta{1};
        opt.test_file = data_meta{2};
        opt.pl = data_meta{3};
        opt.nl = data_meta{4};
        %random seed
        opt.seed = round;
        %be careful that data_id can't include '.' !
        data_id = ['btpr_result/btpr_', data_meta{5}, 'svmpauc_result'];
        
        [Xtr, ytr, Xte, yte] = dataPreprocess(opt);
        
        %upper bound and lower bound of tolrance
        u_tol =  0.1;
        
        if size(yte, 1) < 100
            l_tol = 1/floor(size(ytr, 1) * 0.5);
        else
            l_tol = 1/floor(size(ytr, 1) *0.8);
        end
        
        %l_tol = 1/floor(size(ytr, 1) *0.8);
        
        %% List of tolerances
        ts = [u_tol];
        cur_tol = u_tol;
        while 1
            cur_tol = cur_tol * 5e-1;
            if cur_tol < l_tol  break;  end
            ts = [cur_tol, ts];
            
            cur_tol = cur_tol * 2e-1;
            if cur_tol < l_tol  break;  end
            ts = [cur_tol, ts];
        end
        
        %ts = [1e-4, 1e-3, 1e-2, 1e-1];
       
        i_tol = 0;
        for  tolerance = ts
            i_tol = i_tol + 1;
	
	fprintf(['Current Time: ',num2str(fix(clock)), '\n']);
        fprintf('round: %d, tol: %f .\n', round, tolerance);
 %% Initialize global variables 
    %% Manual settings
    target = 'BTPR';
    test_data = 'TEST';
    
    %% lab on?
    ASVM = 0;
    SVM = 0;
    BS_SVM = 0;
    CS_SVM = 0;
    LR = 0;
    SVMpAUC = 1;
    TOPPUSH = 0;
    TOPKPUSH_PART = 0;
    TOPKPUSH = 0;
    
    %% grid search on?
    ASVM_GridSearch_ON = 1;
    SVM_GridSearch_ON = 1;
    BS_SVM_GridSearch_ON = 1; 
    CS_SVM_GridSearch_ON = 1;
    LR_GridSearch_ON = 1;
    SVMpAUC_GridSearch_ON = 1;
    TOPPUSH_GridSearch_ON = 1;
    TOPKPUSH_GridSearch_ON = 1;
    
    %time-consuming?
    SAVE_TIME = 1;
    
    %plotting?
    plot_on = 0;
    
    %save_model?
    SAVE_MODEL = 0;

    %liblinear
    liblinear = 1;
    
    if SAVE_TIME
        %reduce n_fold when computation is time-consuming.
        %influence: ASVM, SVM, BS_SVM, CS_SVM, LR.
        if size(yte, 1) < 100
            n_fold = 2;
        else
            n_fold = 5;
        end

        %tolerance for SVM based methods
        e = 0.001;
        %enlarge step_size when computation is time-consuming.
        %influence: ASVM, CS_SVM because their searching complexity is O(n^2).
        step_size = 10.0;
    else
        if size(yte, 1) < 100
            n_fold = 2;
        else
            n_fold = 5;
        end

        %tolerance for SVM based methods.
        e = 0.001;
        step_size = 10.0;
    end
       
    %% Auto settings
    if strcmp(test_data, 'TEST') 
        N = size(yte, 1);
    else
        N = size(ytr, 1);
        yte = ytr;
        Xte = Xtr;
    end
    % scores
    VPDT = zeros(N, 0);
    % label predict
    LPDT = zeros(N, 0);
    METHOD = cell(1, 0);

%tic
tic
%% START LABS
if ASVM == 1
%% ASVM
    asvm_model = [];
    if ~ASVM_GridSearch_ON
        try
            load ((strcat(data_id, '_ASVM_GridSearchResult')));
        catch
            fprintf('ASVM read grid-search result failed!');
            ASVM_GridSearch_ON = 1;
        end
    end
    
    if ASVM_GridSearch_ON
        asvm_opt.init_mu = 1e-3; 
        asvm_opt.mu_factor = step_size; 
        asvm_opt.init_tau = 1e-3; 
        asvm_opt.tau_factor = step_size; 
        
        asvm_opt.n_fold = n_fold;
        asvm_opt.t = tolerance; 
        asvm_opt.target = target; 
        asvm_opt.valid_rate = 1/3;
        asvm_opt.e = e;
        
        [best_mu, best_tau, best_target ] = ASVM_GridSearch(ytr, Xtr,  asvm_opt);
        asvm_model = libsvmtrain(ytr, Xtr, ['-s 5 -t 0 -x ', num2str(best_mu), ' -y ', num2str(best_tau), ' -e ', num2str(e), ' -q']);
        if SAVE_MODEL
            save (strcat(data_id, '_ASVM_GridSearchResult'), 'asvm_model'); 
        end
    end
    
    [lpdt, ~, vpdt] = libsvmpredict(yte, Xte, asvm_model);
    
    %update global result
    LPDT = [LPDT, lpdt];
    VPDT = [VPDT, vpdt];
    METHOD = [METHOD, 'ASVM'];
end
toc
tic
if SVM == 1
%% SVM
    svm_model = [];
    svm_liblinear = 0;
    if ~SVM_GridSearch_ON
        try
            load ((strcat(data_id, '_SVM_GridSearchResult')));
        catch
            fprintf('SVM read grid-search result failed!');
            SVM_GridSearch_ON = 1;
        end
    end
    
    if SVM_GridSearch_ON
        svm_opt.c_lb = 1e-3;
        svm_opt.c_ub = 1e3; 
        svm_opt.c_factor = step_size; 
        
        svm_opt.n_fold = n_fold; 
        svm_opt.target = target;
        svm_opt.t = tolerance;
        %this option is only used when n_fold == 1
        svm_opt.valid_rate = 1/3;
        svm_opt.e = e;
        svm_opt.liblinear = liblinear;
        [best_c, best_target ] = SVM_GridSearch( ytr, Xtr,  svm_opt);
        
        if liblinear
            svm_model = liblineartrain(ytr, Xtr, ['-s 3 -c ', num2str(best_c), ' -e ', num2str(e), ' -B 1 -q']);
        else
            svm_model = libsvmtrain(ytr, Xtr, ['-s 0 -t 0 -c ', num2str(best_c), ' -e ', num2str(e)]);
        end
        svm_liblinear = liblinear;
        if SAVE_MODEL
            save (strcat(data_id, '_SVM_GridSearchResult'), 'svm_model', 'svm_liblinear'); 
        end
    end

    %model is loaded from model file succesfully, predict directly.
    if svm_liblinear
        [lpdt, ~, vpdt] = liblinearpredict(yte, Xte, svm_model);
    else
        [lpdt, ~, vpdt] = libsvmpredict(yte, Xte, svm_model);
    end
      
    %update global result
    LPDT = [LPDT, lpdt];
    VPDT = [VPDT, vpdt];
    METHOD = [METHOD, 'SVM'];
end
toc
tic
if BS_SVM == 1
%% BS_SVM
    bs_svm_model = [];
    bs_svm_liblinear = 0;
    if  ~BS_SVM_GridSearch_ON
        try
            load ((strcat(data_id, '_BS_SVM_GridSearchResult')));
        catch
            fprintf('BS_SVM read grid-search result failed!');
            BS_SVM_GridSearch_ON = 1;
        end
    end
    
    if BS_SVM_GridSearch_ON
        %% first train a SVM model focus on AUC, return best hyper-parameters
        svm_opt.c_lb = 1e-3;
        svm_opt.c_ub = 1e3; 
        svm_opt.c_factor = step_size;   
        svm_opt.n_fold = n_fold; 
        svm_opt.target = 'AUC';
        %this option is only used when n_fold == 1
        svm_opt.valid_rate = 1/3;
        svm_opt.e = e;
        svm_opt.liblinear = liblinear;
        [best_c, best_target] = SVM_GridSearch( ytr, Xtr,  svm_opt);  
        
        %% then using selected hyper-parameters above to train a SVM model and select conform bias by specified target
        bs_svm_opt.t = tolerance;
        %this parameter only used when fulltrain = 0;
        bs_svm_opt.valid_rate = 1/3;
        bs_svm_opt.target = 'NP_SCORE';
        bs_svm_opt.is_fulltrain = 0;
        bs_svm_opt.c = best_c;
        bs_svm_opt.e = e;
        bs_svm_opt.liblinear = liblinear;
        bs_svm_model = BS_SVM_ThresholdSelect(ytr, Xtr, bs_svm_opt);
        bs_svm_liblinear = liblinear;
        if SAVE_MODEL
            save (strcat(data_id, '_BS_SVM_GridSearchResult'), 'bs_svm_model', 'bs_svm_liblinear'); 
        end
    end
    
    if bs_svm_liblinear
        [lpdt, ~, vpdt] = liblinearpredict(yte, Xte, bs_svm_model);
    else
        [lpdt, ~, vpdt] = libsvmpredict(yte, Xte, bs_svm_model);
    end
    
    %update global result
    LPDT = [LPDT, lpdt];
    VPDT = [VPDT, vpdt];
    METHOD = [METHOD, 'BS_SVM'];
end
toc
tic    
if CS_SVM == 1
%% CS_SVM
    cs_svm_model = [];
    cs_svm_liblinear = 0;
    if ~CS_SVM_GridSearch_ON
        try
            load ((strcat(data_id, '_CS_SVM_GridSearchResult')));
        catch
            fprintf('CS_SVM read grid-search result failed!');
            CS_SVM_GridSearch_ON = 1;
        end
    end
    
    if CS_SVM_GridSearch_ON
        cs_svm_opt.n_fold = n_fold;
        cs_svm_opt.target = target;
        cs_svm_opt.t = tolerance;
        %this option is only used when n_fold == 1
        cs_svm_opt.valid_rate = 1/3;
        cs_svm_opt.e = e;
        cs_svm_opt.liblinear = liblinear;
        
        cs_svm_opt.c_lb = 1e-3;
        cs_svm_opt.c_ub = 1e3;
        cs_svm_opt.c_factor = step_size;
        %w = C+/C-, w <= 1 is suitable for low fp. 
        cs_svm_opt.w_lb = 1e-3;
        cs_svm_opt.w_ub = 1;
        cs_svm_opt.w_factor = step_size;
        [best_c, best_w, best_target ] = CS_SVM_GridSearch( ytr, Xtr,  cs_svm_opt);
        
        if liblinear
            cs_svm_model = liblineartrain(ytr, Xtr, ['-s 3 -c ', num2str(best_c), ' -w1 ', num2str(best_w), ' -e ', num2str(e), ' -B 1 -q']);
        else
            cs_svm_model = libsvmtrain(ytr, Xtr, ['-s 0 -t 0 -c ', num2str(best_c), ' -w1 ', num2str(best_w), ' -e ', num2str(e)]);
        end
        cs_svm_liblinear = liblinear;
        if SAVE_MODEL
            save (strcat(data_id, '_CS_SVM_GridSearchResult'), 'cs_svm_model', 'cs_svm_liblinear'); 
        end
    end

    %model is loaded from model file succesfully, predict directly.
    if cs_svm_liblinear
        [lpdt, ~, vpdt] = liblinearpredict(yte, Xte, cs_svm_model);
    else
        [lpdt, ~, vpdt] = libsvmpredict(yte, Xte, cs_svm_model);
    end
    
    %update global result
    LPDT = [LPDT, lpdt];
    VPDT = [VPDT, vpdt];
    METHOD = [METHOD, 'CS_SVM'];
end
toc
tic
if LR == 1
%% LR
    lr_model = [];
    if ~LR_GridSearch_ON
        try
            load ((strcat(data_id, '_LR_GridSearchResult')));
        catch
            fprintf('LR read grid-search result failed!');
            LR_GridSearch_ON = 1;
        end
    end
    
    if LR_GridSearch_ON
        lr_opt.n_fold = n_fold;
        lr_opt.target = target;
        lr_opt.t = tolerance;
        %this option is only used when n_fold == 1
        lr_opt.valid_rate = 1/3;
        
        lr_opt.c_lb = 1e-3;
        lr_opt.c_ub = 1e3;
        lr_opt.c_factor = step_size;
    
        [best_c, best_target ] = LR_GridSearch(ytr, Xtr,  lr_opt);
        lr_model = liblineartrain(ytr, Xtr, ['-s 0 -c ', num2str(best_c), ' -q']);
        if SAVE_MODEL
            save (strcat(data_id, '_LR_GridSearchResult'), 'lr_model'); 
        end
    end
    [lpdt, ~, vpdt] = liblinearpredict(yte, Xte, lr_model);
    
    %update global result
    LPDT = [LPDT, lpdt];
    VPDT = [VPDT, vpdt];
    METHOD = [METHOD, 'LR'];
end
toc
tic
if SVMpAUC == 1
%% TopPush GridSearch
    best_lambda = inf;
    if ~SVMpAUC_GridSearch_ON
        try
            load ((strcat(data_id, 'SVMpAUC_GridSearchResult')));
        catch
            fprintf('SVMpAUC read grid-search result failed!');
            SVMpAUC_GridSearch_ON = 1; 
        end
    end
    
    if SVMpAUC_GridSearch_ON
        opt.n_fold = n_fold;
        opt.target = target;
        opt.t = tolerance;
        %this option is only used when n_fold == 1
        opt.valid_rate = 1/3;

        opt.lambda_lb = 1e-3;
        opt.lambda_ub = 1e3;
        opt.lambda_factor = step_size;

        [best_lambda, best_target_0] = SVMpAUC_GridSearch(Xtr, ytr, opt); 
        if SAVE_MODEL
            save (strcat(data_id, 'SVMpAUC_GridSearchResult'), 'best_lambda'); 
        end
    end

    ttt = max(0, tolerance - 1/sum(ytr==-1));
    tt = ttt + 1.01/sum(ytr==-1);
    w = SVMpAUC_TIGHT(Xtr, ytr, ttt, tt, best_lambda, 1e-4, 0);
    pdt = Xte*w;

    %update global result
    VPDT = [VPDT, pdt(:, 1)];
    METHOD = [METHOD, 'SVM_pAUC'];
end
toc
tic
if TOPPUSH == 1
%% TopPush GridSearch
    best_lambda = inf;
    if ~TOPPUSH_GridSearch_ON
        try
            load ((strcat(data_id, 'TOPPUSH_GridSearchResult')));
        catch
            fprintf('TOPPUSH read grid-search result failed!');
            TOPPUSH_GridSearch_ON = 1; 
        end
    end
    
    if TOPPUSH_GridSearch_ON
        oob_opt.n_fold = n_fold;
        oob_opt.target = target;
        oob_opt.t = tolerance;
        %this option is only used when n_fold == 1
        oob_opt.valid_rate = 1/3;

        oob_opt.K_on = 0;
        oob_opt.lambda_lb = 1e-3;
        oob_opt.lambda_ub = 1e3;
        oob_opt.lambda_factor = step_size;
        oob_opt.k = tolerance;

        [best_lambda, ~, best_target_1] = topKPushRank(Xtr, ytr, oob_opt); 
        if SAVE_MODEL
            save (strcat(data_id, 'TOPPUSH_GridSearchResult'), 'best_lambda'); 
        end
    end

    disp(best_lambda);
    %wkr_opt initialization
    wkr_opt.lambda =  best_lambda;
    wkr_opt.maxIter = 10000;
    wkr_opt.tol = 1e-4;
    wkr_opt.debug = false;
    %calculate w
    model = topPushTrain(Xtr, ytr, wkr_opt);
    pdt = modelPredict(model, Xte, 'SCORE');

    %update global result
    VPDT = [VPDT, pdt(:, 1)];
    METHOD = [METHOD, 'TOPPUSH GridSearch'];
end
toc
if TOPKPUSH_PART == 1
%% TopKPush GridSearch
    best_lambda = inf;
    if ~TOPKPUSH_GridSearch_ON
        try
            load ((strcat(data_id, 'TOPKPUSH_GridSearchResult')));
        catch
            fprintf('TOPKPUSH read grid-search result failed!');
            TOPKPUSH_GridSearch_ON = 1; 
        end
    end
    
    if TOPKPUSH_GridSearch_ON
        oob_opt.n_fold = n_fold;
        oob_opt.target = target;
        oob_opt.t = tolerance;
        %this option is only used when n_fold == 1
        oob_opt.valid_rate = 1/3;

        oob_opt.K_on = 1;
        oob_opt.lambda_lb = 1e-3;
        oob_opt.lambda_ub = 1e3;
        oob_opt.lambda_factor = step_size;
        oob_opt.k = tolerance;
        oob_opt.search_k = 1;

        [best_lambda, best_k, best_target_k] = topKPushRank(Xtr, ytr, oob_opt); 
        if SAVE_MODEL
            save (strcat(data_id, 'TOPKPUSH_GridSearchResult'), 'best_lambda'); 
        end
    end

    %wkr_opt initialization
    wkr_opt.lambda =  best_lambda;
    wkr_opt.maxIter = 10000;
    wkr_opt.tol = 1e-4;
    wkr_opt.debug = false;
    wkr_opt.k = best_k;
    
    %calculate w
    model = topKPushPreciseTrain(Xtr, ytr, wkr_opt);
    pdt = modelPredict(model, Xte, 'SCORE');

    %update global result
    VPDT = [VPDT, pdt(:, 1)];
    METHOD = [METHOD, 'TOPKPUSH PART GridSearch'];
end
tic
if TOPKPUSH == 1
%% TopKPush GridSearch
    best_lambda = inf;
    if ~TOPKPUSH_GridSearch_ON
        try
            load ((strcat(data_id, 'TOPKPUSH_GridSearchResult')));
        catch
            fprintf('TOPKPUSH read grid-search result failed!');
            TOPKPUSH_GridSearch_ON = 1; 
        end
    end
    
    if TOPKPUSH_GridSearch_ON
        oob_opt.n_fold = n_fold;
        oob_opt.target = target;
        oob_opt.t = tolerance;
        %this option is only used when n_fold == 1
        oob_opt.valid_rate = 1/3;

        oob_opt.K_on = 1;
        oob_opt.lambda_lb = 1e-3;
        oob_opt.lambda_ub = 1e3;
        oob_opt.lambda_factor = step_size;
        oob_opt.k = tolerance;

        [best_lambda, ~, best_target_k] = topKPushRank(Xtr, ytr, oob_opt); 
        if SAVE_MODEL
            save (strcat(data_id, 'TOPKPUSH_GridSearchResult'), 'best_lambda'); 
        end
    end

    %wkr_opt initialization
    wkr_opt.lambda =  best_lambda;
    wkr_opt.maxIter = 10000;
    wkr_opt.tol = 1e-4;
    wkr_opt.debug = false;
    wkr_opt.k = tolerance;
    
    %calculate w
    model = topKPushPreciseTrain(Xtr, ytr, wkr_opt);
    pdt = modelPredict(model, Xte, 'SCORE');

    %update global result
    VPDT = [VPDT, pdt(:, 1)];
    METHOD = [METHOD, 'TOPKPUSH GridSearch'];
end
toc
%% Evaluation and Plotting 
    M = size(METHOD, 2);
    auc = zeros(1, M);
    btpr = zeros(1, M);
    %for plotting
    fpr = cell(1, M);
    tpr = cell(1, M);
    
    %for plotting
    color_mat = {'r', 'b', 'g', 'k', 'm'};
    line_mat = {'-', '--', ':', '.-'};
    
    for i = 1: M
        [auc(1, i), fpr{1, i}, tpr{1, i}, btpr(1, i)] = calculate_roc(VPDT(:, i), yte, 1, -1, tolerance);
        
        if plot_on
            %for plotting: consturct curve style
            line_style = [line_mat{1 + floor(i / size(line_mat, 2))}, color_mat{mod(i, size(color_mat, 2)) + 1}];  
            plot(fpr{1, i}, tpr{1, i}, line_style);
            hold on;
        end
    end
    
    if plot_on
        %for plotting
        legend(METHOD);
    end
    
    auc_result{i_tol, round} = auc;
    btpr_result{i_tol, round} = btpr;
        end
    end 
    %save result of this dataset
    save (data_id, 'auc_result', 'btpr_result', 'METHOD');
end
