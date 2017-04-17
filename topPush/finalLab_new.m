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
        {'data/w8a.txt', 'data/w8a.t', 1, -1, 'w8a'}, ...
        {'data/ijcnn1', 'data/ijcnn1.t', 1, -1, 'ijcnn'}, ...
    };

datasets = {
        {'data/colon-cancer', '', 1, -1, 'colon-cancer'}, ...
        {'data/diabetes.txt', '', 1, -1, 'diabetes'}, ...
        {'data/german.numer.txt', '', 1, -1, 'german'}, ...
        {'data/heart.txt', '', 1, -1, 'heart'}, ...
        {'data/ionosphere.txt', '', 1, -1, 'ionosphere'}, ...
        {'data/leu', 'data/leu.t', 1, -1, 'leu'}, ...
        {'data/liver-disorders.txt', 'data/liver-disorders.t', 1, 0, 'liver-disorders'}, ...
        {'data/a8a.txt', 'data/a8a.t', 1, -1, 'a8a'}, ...
        {'data/news20.binary', '', 1, -1, 'news20'}, ...
        {'data/w8a.txt', 'data/w8a.t', 1, -1, 'w8a'}, ...
        {'data/real-sim', '', 1, -1, 'real-sim'}, ...
	{'data/rcv1_train.binary', 'data/rcv1_test.binary', 1, -1, 'rcv1'}, ...
        {'data/ijcnn1', 'data/ijcnn1.t', 1, -1, 'ijcnn'}, ...
    };


datasets =  {
	{'data/australian.txt', '', 1, -1, 'australian'}, ...
        {'data/breast-cancer.txt', '', 2, 4, 'breast_cancer'}, ...
	{'data/colon-cancer', '', 1, -1, 'colon-cancer'}, ...
        {'data/diabetes.txt', '', 1, -1, 'diabetes'}, ...
        {'data/german.numer.txt', '', 1, -1, 'german'}, ...
        {'data/heart.txt', '', 1, -1, 'heart'}, ...
        {'data/ionosphere.txt', '', 1, -1, 'ionosphere'}, ...
        {'data/leu', 'data/leu.t', 1, -1, 'leu'}, ...
        {'data/liver-disorders.txt', 'data/liver-disorders.t', 1, 0, 'liver-disorders'}, ...
    };
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
    {'data/real-sim', '', 1, -1, 'real-sim'}, ... 
    {'data/a8a.txt', 'data/a8a.t', 1, -1, 'a8a'}, ... 
    {'data/news20.binary', '', 1, -1, 'news20'}, ...
    {'data/w8a.txt', 'data/w8a.t', 1, -1, 'w8a'}, ...
    {'data/rcv1_train.binary', 'data/rcv1_test.binary', 1, -1, 'rcv1'}, ...
};

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
};

datasets = {
	{'data/covtype.libsvm.binary.scale', '', 2, 1, 'covtype'},...	
};

datasets = {
	{'data/SUSY', '', 1, -1, 'covtype'}
}
%% Read data into memory
    %For data file
for  data_cell = datasets  
    auc_result = cell(0, 0);
    btpr_result = cell(0, 0);
    np_result = cell(0, 0);
    fpr_result = cell(0, 0);
    tpr_result = cell(0, 0);
    
    for round = 1:1
        data_meta = data_cell{1};
        opt.train_file = data_meta{1};
        opt.test_file = data_meta{2};
        opt.pl = data_meta{3};
        opt.nl = data_meta{4};
        %random seed
        opt.seed = round;
        %be careful that data_id can't include '.' !
        data_id = ['large_result/', data_meta{5}, '_result'];
        
        %%SUSY!
        opt.susy = 1;

        [Xtr, ytr, Xte, yte] = dataPreprocess(opt);
        disp(size(Xtr));
disp(size(ytr));
disp(size(Xte));
disp(size(yte));    
        %upper bound and lower bound of tolrance
        u_tol =  0.1;
        l_tol = 1e-4;
        
        %% List of tolerances
        ts = [u_tol];
        c_tol = u_tol;
        te_size = size(yte, 1);

        while 1
            c_tol = c_tol * 5e-1;
            if c_tol < 1e-4
                break;
            end
            ts = [c_tol, ts];
            if ceil(c_tol * te_size)  == 1
                break;
            end
            
            c_tol = c_tol * 2e-1;
            if c_tol < 1e-4
                break;
            end
            ts = [c_tol, ts];
            if ceil(c_tol * te_size)  == 1
                break;
            end
        end


        
        ts = [1e-2, 1e-3, 0.1, 0.01];
       
        i_tol = 0;
        for  tolerance = ts
            i_tol = i_tol + 1;
            
        fprintf(['Current Time: ',num2str(fix(clock)), '\n']);
        fprintf('round: %d, tol: %f .\n', round, tolerance);
    %be careful that data_id can't include '.' !
    %data_id = [data_meta{5}, '_', regexprep(num2str(tolerance), '\.', ''), '_', num2str(round)];

 %% Initialize global variables 
    %% Manual settings
    target = 'NP_SCORE'; 
    test_data = 'TEST';
    
    %% lab on?
    ASVM = 0;
    SVM = 1;
    BS_SVM = 1;
    CS_SVM = 1;
    LR = 1;
    TOPPUSH = 0;
    OOB_HM = 0;
    KOOB_HM = 0;
    OOB_SM = 1;
    KOOB_SM = 0;
    OOB_HS = 0;
    OOB_SS = 1;
    PKOOB_PART_HM = 0;
    PKOOB_PART_SM = 0;
    PKOOB_HS = 0;
    PKOOB_SS = 0;
    PKOOB_HM = 0;
    PKOOB_SM = 1;
    
    %% grid search on?
    ASVM_GridSearch_ON = 1;
    SVM_GridSearch_ON = 1;
    BS_SVM_GridSearch_ON = 1; 
    CS_SVM_GridSearch_ON = 1;
    LR_GridSearch_ON = 1;
    OOB_GridSearch_HM_ON = 1;
    OOB_GridSearch_SM_ON = 1;
    KOOB_GridSearch_HM_ON = 1;
    KOOB_GridSearch_SM_ON = 1;
    OOB_GridSearch_HS_ON = 1;
    OOB_GridSearch_SS_ON = 1;
    PKOOB_PART_GridSearch_HM_ON = 1;
    PKOOB_PART_GridSearch_SM_ON = 1;
    PKOOB_GridSearch_HS_ON = 1;
    PKOOB_GridSearch_SS_ON = 1;
    PKOOB_GridSearch_HM_ON = 1;
    PKOOB_GridSearch_SM_ON = 1;
 
%     %time-consuming?
%     if size(ytr, 1) < 10000
%         SAVE_TIME = 0;
%     else
%         SAVE_TIME = 1;
%     end
    
    %time-consuming?
    SAVE_TIME = 1;
    
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
        %bn for OOB based methods
        bn = 10;
    else
        if size(yte, 1) < 100
            n_fold = 2;
        else
            n_fold = 5;
        end
        %tolerance for SVM based methods.
        e = 0.001;
        step_size = 10.0;
        bn = 10;
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
    %fixed random seed
    %rng(100);

%tic

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
        asvm_model = libsvmtrain(ytr, Xtr, ['-s 5 -t 0 -x ', num2str(best_mu), ' -y ', num2str(best_tau), ' -e ', num2str(e)]);
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

if TOPPUSH == 1
%% Naive TOPPUSH
    toppush_opt.lambda = 1;
    toppush_opt.maxIter = 10000;
    toppush_opt.tol = 1e-4;
    toppush_opt.debug = false;

    model = topPushTrain(Xtr, ytr, toppush_opt);
    vpdt = modelPredict(model, Xte, 'SCORE');
    
    %update global result
    LPDT = [LPDT, zeros(N, 1)];
    VPDT = [VPDT, vpdt];
    METHOD = [METHOD, 'TOPPUSH'];
end

tic 
if OOB_HM == 1
%% TopPush OOB HARD MERGE GridSearch
    best_oob_model = [];
    if ~OOB_GridSearch_HM_ON
        try
            load ((strcat(data_id, '_OOB_H_GridSearchResult')));
        catch
            fprintf('OOB HM read grid-search result failed!');
            OOB_GridSearch_HM_ON = 1;
        end
    end
    
    if OOB_GridSearch_HM_ON
        oob_opt.tolerance = tolerance;  
        oob_opt.bn = bn;
        oob_opt.sr = 2/3;
        oob_opt.fsr = 1; 
        oob_opt.seed = 'dynamic';
        oob_opt.TH_method = 'hard';
        oob_opt.oob_method = 'default'; 
        oob_opt.wk_learner_train = @topPushTrain;
        oob_opt.lambda_lb = 1e-3;
        oob_opt.lambda_ub = 1e3;
        oob_opt.lambda_factor = step_size;
    
        [best_oob_model, best_lambda, best_btpr] = OOB_GridSearch(Xtr, ytr, oob_opt); 
        if SAVE_MODEL
            save (strcat(data_id, '_OOB_H_GridSearchResult'), 'best_oob_model', 'best_lambda'); 
        end
    end

    %calculate real b and w
    model.name = 'TOPPUSH';
    model.w = mean(best_oob_model.w, 2);
    model.b = mean(best_oob_model.b);
    pdt = modelPredict(model, Xte, 'ALL');
    
    %update global result
    LPDT = [LPDT, pdt(:, 2)];
    VPDT = [VPDT, pdt(:, 1)];
    METHOD = [METHOD, 'TOP PUSH OOB HARD MERGE GridSearch'];
end
toc

if KOOB_HM == 1
%% TopKPush OOB HARD MERGE GridSearch
    best_oob_model = [];
    if ~KOOB_GridSearch_HM_ON
        try
            load ((strcat(data_id, '_KOOB_HM_GridSearchResult')));
        catch
            fprintf('KOOB HM read grid-search result failed!');
            KOOB_GridSearch_HM_ON = 1;
        end
    end
    
    if KOOB_GridSearch_HM_ON
        oob_opt.tolerance = tolerance;  
        oob_opt.bn = bn;
        oob_opt.sr = 2/3;
        oob_opt.fsr = 1; 
        oob_opt.seed = 'dynamic';
        oob_opt.TH_method = 'hard';
        oob_opt.oob_method = 'default'; 
        oob_opt.wk_learner_train = @topKPushTrain;
        oob_opt.lambda_lb = 1e-3;
        oob_opt.lambda_ub = 1e3;
        oob_opt.lambda_factor = step_size;
        oob_opt.k = tolerance;
    
        [best_oob_model, best_lambda, best_btpr] = OOB_GridSearch(Xtr, ytr, oob_opt); 
        save (strcat(data_id, '_KOOB_HM_GridSearchResult'), 'best_oob_model', 'best_lambda'); 
    end

    %calculate real b and w
    model.name = 'TOPPUSH';
    model.w = mean(best_oob_model.w, 2);
    model.b = mean(best_oob_model.b);
    pdt = modelPredict(model, Xte, 'ALL');

    %update global result
    LPDT = [LPDT, pdt(:, 2)];
    VPDT = [VPDT, pdt(:, 1)];
    METHOD = [METHOD, 'TOP K PUSH OOB HARD MERGE GridSearch'];
end
 
tic
if OOB_SM == 1
%% TopPush OOB SOFT MERGE GridSearch
    best_oob_model = [];
    if ~OOB_GridSearch_SM_ON
        try
            load ((strcat(data_id, '_OOB_S_GridSearchResult')));
        catch
            fprintf('OOB SM read grid-search result failed!');
            OOB_GridSearch_SM_ON = 1;
        end
    end
    
    if OOB_GridSearch_SM_ON
        oob_opt.tolerance = tolerance;  
        oob_opt.bn = bn;
        oob_opt.sr = 2/3;
        oob_opt.fsr = 1; 
        oob_opt.seed = 'dynamic';
        oob_opt.TH_method = 'soft';
        oob_opt.oob_method = 'default'; 
        oob_opt.wk_learner_train = @topPushTrain;
        oob_opt.lambda_lb = 1e-3;
        oob_opt.lambda_ub = 1e3;
        oob_opt.lambda_factor = step_size;

        [best_oob_model, best_lambda, best_btpr] = OOB_GridSearch(Xtr, ytr, oob_opt); 
        if SAVE_MODEL
            save (strcat(data_id, '_OOB_S_GridSearchResult'), 'best_oob_model', 'best_lambda'); 
        end
    end
    
    %calculate real b and w
    model.name = 'TOPPUSH';
    model.w = mean(best_oob_model.w, 2);
    model.b = mean(best_oob_model.b);
    pdt = modelPredict(model, Xte, 'ALL');
    
    %update global result
    LPDT = [LPDT, pdt(:, 2)];
    VPDT = [VPDT, pdt(:, 1)];
    METHOD = [METHOD, 'TOP PUSH OOB SOFT MERGE GridSearch'];
end
toc

if KOOB_SM == 1
%% TopKPush OOB SOFT MERGE GridSearch
    best_oob_model = [];
    if ~KOOB_GridSearch_SM_ON
        try
            load ((strcat(data_id, '_KOOB_SM_GridSearchResult')));
        catch
            fprintf('KOOB SM read grid-search result failed!');
            KOOB_GridSearch_SM_ON = 1; 
        end
    end
    
    if KOOB_GridSearch_SM_ON
        oob_opt.tolerance = tolerance;  
        oob_opt.bn = bn;
        oob_opt.sr = 2/3;
        oob_opt.fsr = 1; 
        oob_opt.seed = 'dynamic';
        oob_opt.TH_method = 'soft';
        oob_opt.oob_method = 'default'; 
        oob_opt.wk_learner_train = @topKPushTrain;
        oob_opt.lambda_lb = 1e-3;
        oob_opt.lambda_ub = 1e3;
        oob_opt.lambda_factor = step_size;
        oob_opt.k = tolerance;

        [best_oob_model, best_lambda, best_btpr] = OOB_GridSearch(Xtr, ytr, oob_opt); 
        save (strcat(data_id, '_KOOB_SM_GridSearchResult'), 'best_oob_model', 'best_lambda'); 
    end
   
    %calculate real b and w
    model.name = 'TOPPUSH';
    model.w = mean(best_oob_model.w, 2);
    model.b = mean(best_oob_model.b);
    pdt = modelPredict(model, Xte, 'ALL');
    
    %update global result
    LPDT = [LPDT, pdt(:, 2)];
    VPDT = [VPDT, pdt(:, 1)];
    METHOD = [METHOD, 'TOP K PUSH OOB SOFT MERGE GridSearch'];
end

tic
if OOB_HS == 1
%% TopPush OOB HARD SINGLE GridSearch
    best_oob_model = [];
    best_lambda = inf;
    if ~OOB_GridSearch_HS_ON
        try
            load ((strcat(data_id, '_OOB_H_GridSearchResult')));
        catch
            fprintf('OOB HS read grid-search result failed!');
            OOB_GridSearch_HS_ON = 1; 
        end
    end
    
    if OOB_GridSearch_HS_ON
        oob_opt.tolerance = tolerance;  
        oob_opt.bn = bn;
        oob_opt.sr = 2/3;
        oob_opt.fsr = 1; 
        oob_opt.seed = 'dynamic';
        oob_opt.TH_method = 'hard';
        oob_opt.oob_method = 'default'; 
        oob_opt.wk_learner_train = @topPushTrain;
        oob_opt.lambda_lb = 1e-3;
        oob_opt.lambda_ub = 1e3;
        oob_opt.lambda_factor = step_size;

        [best_oob_model, best_lambda, best_btpr] = OOB_GridSearch(Xtr, ytr, oob_opt); 
        if SAVE_MODEL
            save (strcat(data_id, '_OOB_H_GridSearchResult'), 'best_oob_model', 'best_lambda'); 
        end
    end

    %wkr_opt initialization
    wkr_opt.lambda =  best_lambda;
    wkr_opt.maxIter = 10000;
    wkr_opt.tol = 1e-4;
    wkr_opt.debug = false;
    %calculate real b and w
    model = topPushTrain(Xtr, ytr, wkr_opt);
    model.b = mean(best_oob_model.b);

    pdt = modelPredict(model, Xte, 'ALL');

    %update global result
    LPDT = [LPDT, pdt(:, 2)];
    VPDT = [VPDT, pdt(:, 1)];
    METHOD = [METHOD, 'TOP PUSH OOB HARD SINGLE GridSearch'];
end
toc

tic
if OOB_SS == 1
%% TopPush OOB SOFT SINGLE GridSearch
    best_oob_model = [];
    best_lambda = inf;  
    if ~OOB_GridSearch_SS_ON
        try
            load ((strcat(data_id, '_OOB_S_GridSearchResult')));
        catch
            fprintf('OOB SS read grid-search result failed!');
            OOB_GridSearch_SS_ON = 1;
        end
    end
    
    if OOB_GridSearch_SS_ON
        oob_opt.tolerance = tolerance;  
        oob_opt.bn = bn;
        oob_opt.sr = 2/3;
        oob_opt.fsr = 1; 
        oob_opt.seed = 'dynamic';
        oob_opt.TH_method = 'soft';
        oob_opt.oob_method = 'default'; 
        oob_opt.wk_learner_train = @topPushTrain;
        oob_opt.lambda_lb = 1e-3;
        oob_opt.lambda_ub = 1e3;
        oob_opt.lambda_factor = step_size;
        
        [best_oob_model, best_lambda, best_btpr] = OOB_GridSearch(Xtr, ytr, oob_opt); 
        if SAVE_MODEL
            save (strcat(data_id, '_OOB_S_GridSearchResult'), 'best_oob_model', 'best_lambda'); 
        end
    end

    %wkr_opt initialization
    wkr_opt.lambda =  best_lambda;
    wkr_opt.maxIter = 10000;
    wkr_opt.tol = 1e-4;
    wkr_opt.debug = false;
    %calculate real b and w
    model = topPushTrain(Xtr, ytr, wkr_opt);
    model.b = mean(best_oob_model.b);
    pdt = modelPredict(model, Xte, 'ALL');

    %update global result
    LPDT = [LPDT, pdt(:, 2)];
    VPDT = [VPDT, pdt(:, 1)];
    METHOD = [METHOD, 'TOP PUSH OOB SOFT SINGLE GridSearch'];
end
toc

tic
if PKOOB_PART_HM == 1
%% TopKPrecisePush OOB HARD MERGE GridSearch
    best_oob_model = [];
    if ~PKOOB_PART_GridSearch_HM_ON
        try
            load ((strcat(data_id, '_PKOOB_PART_HM_GridSearchResult')));
        catch
            fprintf('PKOOB PART HM read grid-search result failed!');
            PKOOB_PART_GridSearch_HM_ON = 1;
        end
    end
    
    if PKOOB_PART_GridSearch_HM_ON
        oob_opt.tolerance = tolerance;  
        oob_opt.bn = bn;
        oob_opt.sr = 2/3;
        oob_opt.fsr = 1; 
        oob_opt.seed = 'dynamic';
        oob_opt.TH_method = 'hard';
        oob_opt.oob_method = 'default'; 
        oob_opt.wk_learner_train = @topKPushPreciseTrain;
        oob_opt.lambda_lb = 1e-3;
        oob_opt.lambda_ub = 1e3;
        oob_opt.lambda_factor = step_size;
        oob_opt.k = tolerance*2;
    
        [best_oob_model, best_lambda, best_btpr] = OOB_GridSearch(Xtr, ytr, oob_opt); 
        if SAVE_MODEL
            save (strcat(data_id, '_PKOOB_PART_HM_GridSearchResult'), 'best_oob_model', 'best_lambda'); 
        end
    end

    %calculate real b and w
    model.name = 'TOPPUSH';
    model.w = mean(best_oob_model.w, 2);
    model.b = mean(best_oob_model.b);
    pdt = modelPredict(model, Xte, 'ALL');

    %update global result
    LPDT = [LPDT, pdt(:, 2)];
    VPDT = [VPDT, pdt(:, 1)];
    METHOD = [METHOD, 'TOP K PRECISE PUSH OOB HARD MERGE PART GridSearch'];
end
toc

tic
if PKOOB_PART_SM == 1
%% TopKPrecisePush OOB SOFT MERGE  PART GridSearch
    best_oob_model = [];
    if ~PKOOB_PART_GridSearch_SM_ON
        try
            load ((strcat(data_id, '_PKOOB_SM_PART_GridSearchResult')));
        catch
            fprintf('PKOOB SM PART read grid-search result failed!');
            PKOOB_GridSearch_SM_ON = 1; 
        end
    end
    
    if PKOOB_GridSearch_SM_ON
        oob_opt.tolerance = tolerance;  
        oob_opt.bn = bn;
        oob_opt.sr = 2/3;
        oob_opt.fsr = 1; 
        oob_opt.seed = 'dynamic';
        oob_opt.TH_method = 'soft';
        oob_opt.oob_method = 'default'; 
        oob_opt.wk_learner_train = @topKPushPreciseTrain;
        oob_opt.lambda_lb = 1e-3;
        oob_opt.lambda_ub = 1e3;
        oob_opt.lambda_factor = step_size;
        oob_opt.k = tolerance*2;

        [best_oob_model, best_lambda, best_btpr] = OOB_GridSearch(Xtr, ytr, oob_opt); 
        if SAVE_MODEL
            save (strcat(data_id, '_PKOOB_SM_PART_GridSearchResult'), 'best_oob_model', 'best_lambda'); 
        end
    end
   
    %calculate real b and w
    model.name = 'TOPPUSH';
    model.w = mean(best_oob_model.w, 2);
    model.b = mean(best_oob_model.b);
    pdt = modelPredict(model, Xte, 'ALL');
    
    %update global result
    LPDT = [LPDT, pdt(:, 2)];
    VPDT = [VPDT, pdt(:, 1)];
    METHOD = [METHOD, 'TOP K PRECISE PUSH OOB SOFT MERGE PART GridSearch'];
end
toc

tic
if PKOOB_HS == 1
%% TopKPrecisePush OOB HARD Single GridSearch
    best_oob_model = [];
    if ~PKOOB_GridSearch_HS_ON
        try
            load ((strcat(data_id, '_PKOOB_HS_GridSearchResult')));
        catch
            fprintf('PKOOB HS read grid-search result failed!');
            PKOOB_GridSearch_HS_ON = 1;
        end
    end
    
    if PKOOB_GridSearch_HS_ON
        oob_opt.tolerance = tolerance;  
        oob_opt.bn = bn;
        oob_opt.sr = 2/3;
        oob_opt.fsr = 1; 
        oob_opt.seed = 'dynamic';
        oob_opt.TH_method = 'hard';
        oob_opt.oob_method = 'default'; 
        oob_opt.wk_learner_train = @topKPushPreciseTrain;
        oob_opt.lambda_lb = 1e-3;
        oob_opt.lambda_ub = 1e3;
        oob_opt.lambda_factor = step_size;
        oob_opt.k = tolerance;
    
        [best_oob_model, best_lambda, best_btpr] = OOB_GridSearch(Xtr, ytr, oob_opt); 
        if SAVE_MODEL
            save (strcat(data_id, '_PKOOB_HS_GridSearchResult'), 'best_oob_model', 'best_lambda'); 
        end
    end

    wkr_opt.lambda =  best_lambda;
    wkr_opt.maxIter = 10000;
    wkr_opt.tol = 1e-4;
    wkr_opt.debug = false;
    wkr_opt.k = tolerance;
    %calculate real b and w
    model = topKPushPreciseTrain(Xtr, ytr, wkr_opt);
    model.b = mean(best_oob_model.b);
    pdt = modelPredict(model, Xte, 'ALL');

    %update global result
    LPDT = [LPDT, pdt(:, 2)];
    VPDT = [VPDT, pdt(:, 1)];
    METHOD = [METHOD, 'TOP K PRECISE PUSH OOB HARD SINGLE GridSearch'];
end
toc

tic
if PKOOB_SS == 1
%% TopKPrecisePush OOB HARD Single GridSearch
    best_oob_model = [];
    if ~PKOOB_GridSearch_SS_ON
        try
            load ((strcat(data_id, '_PKOOB_SS_GridSearchResult')));
        catch
            fprintf('PKOOB SS read grid-search result failed!');
            PKOOB_GridSearch_SS_ON = 1;
        end
    end
    
    if PKOOB_GridSearch_SS_ON
        oob_opt.tolerance = tolerance;  
        oob_opt.bn = bn;
        oob_opt.sr = 2/3;
        oob_opt.fsr = 1; 
        oob_opt.seed = 'dynamic';
        oob_opt.TH_method = 'soft';
        oob_opt.oob_method = 'default'; 
        oob_opt.wk_learner_train = @topKPushPreciseTrain;
        oob_opt.lambda_lb = 1e-3;
        oob_opt.lambda_ub = 1e3;
        oob_opt.lambda_factor = step_size;
        oob_opt.k = tolerance;
    
        [best_oob_model, best_lambda, best_btpr] = OOB_GridSearch(Xtr, ytr, oob_opt); 
        if SAVE_MODEL
            save (strcat(data_id, '_PKOOB_SS_GridSearchResult'), 'best_oob_model', 'best_lambda'); 
        end
    end

    wkr_opt.lambda =  best_lambda;
    wkr_opt.maxIter = 10000;
    wkr_opt.tol = 1e-4;
    wkr_opt.debug = false;
    wkr_opt.k = tolerance;
    %calculate real b and w
    model = topKPushPreciseTrain(Xtr, ytr, wkr_opt);
    model.b = mean(best_oob_model.b);
    pdt = modelPredict(model, Xte, 'ALL');

    %update global result
    LPDT = [LPDT, pdt(:, 2)];
    VPDT = [VPDT, pdt(:, 1)];
    METHOD = [METHOD, 'TOP K PRECISE PUSH OOB SOFT SINGLE GridSearch'];
end
toc

tic
if PKOOB_HM == 1
%% TopKPrecisePush OOB HARD MERGE GridSearch
    best_oob_model = [];
    if ~PKOOB_GridSearch_HM_ON
        try
            load ((strcat(data_id, '_PKOOB_HM_GridSearchResult')));
        catch
            fprintf('PKOOB HM read grid-search result failed!');
            PKOOB_GridSearch_HM_ON = 1;
        end
    end
    
    if PKOOB_GridSearch_HM_ON
        oob_opt.tolerance = tolerance;  
        oob_opt.bn = bn;
        oob_opt.sr = 2/3;
        oob_opt.fsr = 1; 
        oob_opt.seed = 'dynamic';
        oob_opt.TH_method = 'hard';
        oob_opt.oob_method = 'default'; 
        oob_opt.wk_learner_train = @topKPushPreciseTrain;
        oob_opt.lambda_lb = 1e-3;
        oob_opt.lambda_ub = 1e3;
        oob_opt.lambda_factor = step_size;
        oob_opt.k = tolerance;
    
        [best_oob_model, best_lambda, best_btpr] = OOB_GridSearch(Xtr, ytr, oob_opt); 
        if SAVE_MODEL
            save (strcat(data_id, '_PKOOB_HM_GridSearchResult'), 'best_oob_model', 'best_lambda'); 
        end
    end

    %calculate real b and w
    model.name = 'TOPPUSH';
    model.w = mean(best_oob_model.w, 2);
    model.b = mean(best_oob_model.b);
    pdt = modelPredict(model, Xte, 'ALL');

    %update global result
    LPDT = [LPDT, pdt(:, 2)];
    VPDT = [VPDT, pdt(:, 1)];
    METHOD = [METHOD, 'TOP K PRECISE PUSH OOB HARD MERGE GridSearch'];
end
toc

tic
if PKOOB_SM == 1
%% TopKPrecisePush OOB SOFT MERGE GridSearch
    best_oob_model = [];
    if ~PKOOB_GridSearch_SM_ON
        try
            load ((strcat(data_id, '_PKOOB_SM_GridSearchResult')));
        catch
            fprintf('PKOOB SM read grid-search result failed!');
            PKOOB_GridSearch_SM_ON = 1; 
        end
    end
    
    if PKOOB_GridSearch_SM_ON
        oob_opt.tolerance = tolerance;  
        oob_opt.bn = bn;
        oob_opt.sr = 2/3;
        oob_opt.fsr = 1; 
        oob_opt.seed = 'dynamic';
        oob_opt.TH_method = 'soft';
        oob_opt.oob_method = 'default'; 
        oob_opt.wk_learner_train = @topKPushPreciseTrain;
        oob_opt.lambda_lb = 1e-3;
        oob_opt.lambda_ub = 1e3;
        oob_opt.lambda_factor = step_size;
        oob_opt.k = tolerance;

        [best_oob_model, best_lambda, best_btpr] = OOB_GridSearch(Xtr, ytr, oob_opt); 
        if SAVE_MODEL
            save (strcat(data_id, '_PKOOB_SM_GridSearchResult'), 'best_oob_model', 'best_lambda'); 
        end
    end
   
    %calculate real b and w
    model.name = 'TOPPUSH';
    model.w = mean(best_oob_model.w, 2);
    model.b = mean(best_oob_model.b);
    pdt = modelPredict(model, Xte, 'ALL');
    
    %update global result
    LPDT = [LPDT, pdt(:, 2)];
    VPDT = [VPDT, pdt(:, 1)];
    METHOD = [METHOD, 'TOP K PRECISE PUSH OOB SOFT MERGE GridSearch'];
end
toc

%end time
%toc

%% Evaluation and Plotting 
    M = size(METHOD, 2);
    auc = zeros(1, M);
    btpr = zeros(1, M);
    np_score = zeros(1, M);
    FPR = zeros(1, M);
    TPR = zeros(1, M);
    
    for i = 1: M
        [auc(1, i), ~, ~, btpr(1, i)] = calculate_roc(VPDT(:, i), yte, 1, -1, tolerance);
        [np_score(:, i), FPR(:, i), TPR(:, i)] = neyman_pearson_score(LPDT(:, i), yte, tolerance);
    end
    
    auc_result{i_tol, round} = auc;
    btpr_result{i_tol, round} = btpr;
    np_result{i_tol, round} = np_score;
    fpr_result{i_tol, round} = FPR;
    tpr_result{i_tol, round} = TPR;

save ([data_id,'_',num2str(tolerance)], 'auc_result', 'btpr_result', 'np_result', 'fpr_result', 'tpr_result', 'METHOD');
        end
%save ([data_id,'_',num2str(round)], 'auc_result', 'btpr_result', 'np_result', 'fpr_result', 'tpr_result', 'METHOD');
    end 
    %save result of this dataset
    save (data_id, 'auc_result', 'btpr_result', 'np_result', 'fpr_result', 'tpr_result', 'METHOD');
end
