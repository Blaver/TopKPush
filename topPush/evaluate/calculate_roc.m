function [auc, x, y, btpr] = calculate_roc( predict, ground_truth, pl, nl, tolerance)
%  predict            - �������Բ��Լ��ķ�����
%  ground_truth  - ���Լ�����ȷ��ǩ��������������ֻ���Ƕ����࣬��0��1
%  pl                    -������label
%  nl                    -������label
%  tolerance        -false positive ���̶�
%  auc                  - ���� t-AUC 
%  xx                    - ����fpr����
%  yy                    - ����tpr����
%  btpr                 - ������fpr <= t ��ǰ���£����Դﵽ�����tpr
    [rank_list, Index] = sort(predict);
    sorted_ground_truth = ground_truth(Index);
    [thresholds, ~, ~] = unique(rank_list);
    
    %threshold count
    n_thd = size(thresholds, 1);
    %count of samples
    m = size(ground_truth, 1);
    %count of positive samples
    pos_num = sum(ground_truth == pl);
    %count of negative samples
    neg_num = sum(ground_truth == nl);
    
    %fp rate array
    x = zeros(n_thd + 1, 1);
    %x = zeros(m + 1, 1);
    %tp rate array
    y = zeros(n_thd + 1, 1);
    %y = zeros(m + 1, 1);
    
    x(1) = 1;
    y(1) = 1;
    
    auc = 0;
    btpr = 0;

    TP = pos_num;
    FP = neg_num;
    cur_thd = 0;
    for i = 1:m
        if i == 1 || (i > 1 && rank_list(i) ~= rank_list(i -1))
            cur_thd = cur_thd + 1;
            x(cur_thd) = FP / neg_num;
            y(cur_thd) = TP / pos_num;

            %find largest tpr under the constraint that fpr <= t
            if x(cur_thd) <= tolerance && y(cur_thd) > btpr 
                btpr = y(cur_thd);
            end
            %only if the last fpr <= tolerance, auc can be accumulated.
            if i > 1 && x(cur_thd - 1) <= tolerance
                auc = auc + (y(cur_thd) + y(cur_thd - 1)) * (x(cur_thd - 1) - x(cur_thd)) * 0.5;
            end
        end
        
        if (sorted_ground_truth(i)  == pl)
            TP = TP - 1;
        else 
            FP = FP - 1;
        end
    end
    
    %if allowed, add the last triangle's area. 
    if x(n_thd) <= tolerance
        auc = auc + y(n_thd) * x(n_thd) * 0.5;
    end
end

