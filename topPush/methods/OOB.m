function [ OOBModel ] = OOB( Xtr, ytr, opt)
% Xtr: features
% Ytr: labels
% opt:
%   opt.wkr_opt       -弱分类器的选项参数集
%   opt.tolerance    -false positive 容忍度
%   opt.bn               -oob 数目
%   opt.sr                -采样率
%   opt.fsr               -feature采样率
%

%% parse option parameters
    if ~isfield(opt,'tolerance')	opt.tolerance = 1;     end
    if ~isfield(opt,'bn')	opt.bn = 10;     end
    if ~isfield(opt,'sr')	opt.sr = 2/3;     end
    if ~isfield(opt,'wk_learner_train')	opt.wk_learner_train = @topPushTrain;     end
    if ~isfield(opt,'wkr_opt')	
        default_wkr_opt.lambda =  1;
        opt.wkr_opt = default_wkr_opt;     
    end

%% initlization
    %prevent the situation that t == 0.
    t = max(1e-6, opt.tolerance);
    bn = opt.bn;
    sr = opt.sr;
    wk_learner_train = opt.wk_learner_train;
    wkr_opt = opt.wkr_opt;
    
    %m: number of training samples
    %n: number of original features
    [m, n] = size(Xtr);
    %cache that restore learned weights of each bagger
    W = zeros(n, bn);
    %cache that restore predict scores
    S = zeros(m - ceil(m*sr), bn); 
    %cache that restore label of test set
    L = zeros(m - ceil(m*sr), bn);
    %cache that restore best-true-positive-rates.
    BTPR = zeros(1, bn);
    
%% OOB
    for i = 1: bn
        %% split training data into 2 parts, fix random seed 
        rng(i);
        iA = sort(randperm(m, ceil(m*sr)));
        rng('shuffle');
        iB = setdiff(1:m, iA);
        %training data
        xA = Xtr(iA, :);
        yA = ytr(iA, :);
        %predict data
        xB = Xtr(iB, :);
        yB = ytr(iB, :);

        %% training the ranking model and predict score
        wk_model= wk_learner_train(xA, yA, wkr_opt);
        if isfield(wk_model, 'w')
            W(:, i) = wk_model.w;
        end
        pdt = modelPredict(wk_model, xB, 'SCORE');
        [rk_list, rk_indices] = sort(pdt);
        rk_list_labels = yB(rk_indices);
        
        %cache predicted scores and their real labels
        S(:, i) = rk_list;
        L(:, i) = rk_list_labels;
        
        %%for grid search, evaluate current hyper-parameter by BTPR.
        BTPR(i) = find_BTPR(rk_list_labels, t);
        
%         %% find threshold
%         if strcmp(TH_method, 'hard')
%             thd = find_hard_threshold(rk_list, rk_list_labels, t);
%         elseif strcmp(TH_method, 'soft')
%             thd = find_soft_threshold(rk_list, rk_list_labels, t);
%         end
%         B(i) = thd;
    end
    
%% return final model
%     OOBModel.b = B;
    OOBModel.w = W;
    OOBModel.BTPR = BTPR;
    OOBModel.S = S;
    OOBModel.L = L;
end

%% Finding optimal value of lambda in ranking function, for the best TPR following the tolerance constraint strictly.
function BTPR = find_BTPR(rk_list_labels, t)
    %reverse rk_list_labels for convenience
    rk_list_labels = flip(rk_list_labels);
    
    n = size(rk_list_labels, 1); 
    %false positive tolerance upper bound 
    UB = floor(sum(rk_list_labels == -1) * t);
    
    FP = 0;
    TP = 0;
    for j = 1:n
        %first, update FP  or TP
        if rk_list_labels(j) == -1
            FP = FP + 1;
        else
            TP = TP + 1;
        end
        %then check if FP is overflowed. if so, stop the line search process and accept current TP.
        if FP > UB
            break;
        end
    end
    %calculate and return BTPR
    BTPR = TP/sum(rk_list_labels == 1);
 end
