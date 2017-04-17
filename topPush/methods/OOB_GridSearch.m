function [ best_model, best_lambda, best_target ] = OOB_GridSearch(Xtr, ytr, oob_opt)
%OOB_GRIDSEARCH 
    %% Option parsing and parameter initialization
    if ~isfield(oob_opt,'lambda_lb') oob_opt.lambda_lb = 1e-3; end
    if ~isfield(oob_opt,'lambda_ub') oob_opt.lambda_ub = 1e3; end
    if ~isfield(oob_opt,'lambda_factor') oob_opt.lambda_factor = 3; end
    if ~isfield(oob_opt,'k') oob_opt.k = 0; end
    
    lambda_lb = oob_opt.lambda_lb;
    lambda_ub = oob_opt.lambda_ub;
    lambda_factor = oob_opt.lambda_factor;
    k = oob_opt.k;
    
    best_target = -inf;
    best_lambda = -inf;
    best_model = [];
    
    %these bounds are dynamic.
    dyn_lambda_lb = lambda_lb;
    dyn_lambda_ub = lambda_ub;
    %search direction of lambda
    lambda_sd = 0;
    
    %%start grid search 
    lambda = lambda_lb;
    while  lambda <= dyn_lambda_ub
        %construct wkr_opt 
        wkr_opt.lambda =  lambda;
        wkr_opt.maxIter = 10000;
        wkr_opt.tol = 1e-4;
        wkr_opt.debug = false;
        wkr_opt.k = k;
        
        %train oob model with current lambda, and calculate mean BTPR 
        oob_opt.wkr_opt = wkr_opt;  
        oob_model = OOB(Xtr, ytr, oob_opt);
        BTPR = mean(oob_model.BTPR);
        %is this parameter better?
        if best_target < BTPR
            best_target = BTPR;
            best_lambda = lambda;
            best_model = oob_model;
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
    end
end

