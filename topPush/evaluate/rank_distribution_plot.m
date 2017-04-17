function rank_distribution_plot(pdt_scores,  pdt_names, labels)
% RANK_DISTRIBUTE_PLOT 此处显示有关此函数的摘要。
% pdt_scores: m*n, m 样本个数， n 预测结果数。
% labels: m*1， 样本label。
%   此处显示详细说明

    %show details of testing result
    n = size(pdt_scores, 2); 
    for i = 1: n
        pdt = pdt_scores(:, i); 
        pdt= (pdt - min(pdt))/(max(pdt) - min(pdt));
        ny = pdt(labels == -1);
        py = pdt(labels == 1);

        subplot(n, 1, i);
        h1 = histogram(ny, 50);
        hold on;
        h2 = histogram(py, 50);
      
        h1.BinWidth = 0.005;
        h2.BinWidth = 0.005;
        title(pdt_names(i));
    end
end

