function [ pdt ] = modelPredict( model, X, type)
%MODELPREDICT
    %% Get score 
    if strcmp(model.name, 'TOPPUSH')
        pdt = topPushPredict(model, X);
    end
    
    %% Get label by threshold
    %Only >= threshold will be recognized as positive sample
    %Predict Type£º
    %   'SCORE': return score only;
    %   'LABEL': return label only;
    %   'ALL': return two columns, first column is score, second is label.
    if strcmp(type, 'LABEL')
        pdt = (pdt >= model.b)*2 - 1;
    end
    if strcmp(type, 'ALL')
        pdt = [pdt, (pdt >= model.b)*2 - 1];
    end  
end


%% TOPPUSH Prediction
function [ pdt ] = topPushPredict( model, X )
    pdt = X*model.w;
end