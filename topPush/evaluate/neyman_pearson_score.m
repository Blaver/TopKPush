function [ np_score, FPR, TPR ] = neyman_pearson_score( pdt_label, label, alpha)
%Calcaulate NEYMAN_PEARSON_SCORE 
    alpha = max(alpha, 1e-6);
    FPR = sum(pdt_label(label == -1) == 1) / sum(label == -1);
    TPR = sum(pdt_label(label == 1) == 1) / sum(label == 1);
    np_score =  max(FPR/alpha, 1) - TPR;
end

