function rank_distribution_plot(pdt_scores,  pdt_names, labels)
% RANK_DISTRIBUTE_PLOT �˴���ʾ�йش˺�����ժҪ��
% pdt_scores: m*n, m ���������� n Ԥ��������
% labels: m*1�� ����label��
%   �˴���ʾ��ϸ˵��

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

