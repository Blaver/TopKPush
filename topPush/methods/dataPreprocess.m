function [Xtr, ytr, Xte, yte] = dataPreprocess(opt)
%DATAPREPROCESS
%   opt.train_file ѵ���ļ�·��+����
%   opt.test_file �����ļ�·��+����
%   opt.pl ��������label
%   opt.nl ��������label
%   opt.train_rate û�в����ļ�ʱ������ѵ������������
%   return��
%       ����label�Ѿ��ĳ� +1 -1��ѵ�������Ծ���
%   comments:
%       �ϲ�ѵ���������Լ���������ػ���train test������opt.seed.

    %% Option parsing and parameter initialization
    if ~isfield(opt,'test_file')  opt.test_file = ''; end
    if ~isfield(opt,'train_rate') opt.train_rate = 2/3; end
    if ~isfield(opt,'pl')  opt.pl = 1; end
    if ~isfield(opt,'nl')  opt.nl = -1; end
    if ~isfield(opt,'seed')  opt.seed = 'shuffle'; end
fprintf('start read data\n');
tic    
    train_file = opt.train_file;
    test_file = opt.test_file;
    train_rate = opt.train_rate;
    pl = opt.pl;
    nl = opt.nl;
    seed = opt.seed;
    
    [~, ~, ext] = fileparts(train_file);
    % read mat file, must include 4 variables: [Xtr, ytr, xte, yte].
    if strcmp(ext, '.mat')
        vars = load(train_file);
        Xtr = vars.Xtr;
        ytr = vars.ytr;
        Xte = vars.Xte;
        yte = vars.yte;

        y = [ytr; yte];
        X = [Xtr; Xte];
    % text file, test_file is not null, merge train and test file.
    elseif ~strcmp(test_file, '') 
        [ytr, Xtr] = libsvmread(train_file);
        [yte, Xte] = libsvmread(test_file);
        
        if size(Xte, 2) < size(Xtr, 2)
            Xte = [Xte, zeros(size(Xte, 1), size(Xtr, 2) - size(Xte, 2))];
        elseif size(Xte, 2) > size(Xtr, 2)
            Xtr = [Xtr, zeros(size(Xtr, 1), size(Xte, 2) - size(Xtr, 2))];
        end
        
        y = [ytr; yte];
        X = [Xtr; Xte];
    % only include train file, read it directly.    
    else
        [y, X] = libsvmread(train_file);
    end
    
    rng(seed);
    train_indices = sort(randperm(size(y, 1), ceil(size(y, 1) * train_rate)));
    test_indices = setdiff(1:size(y, 1), train_indices);
    rng('shuffle');

    %change labels
    y(y == pl) = nl + 1;
    y(y == nl) = -1;
    y(y == (nl+ 1)) = 1;

    ytr = y(train_indices, 1);
    yte = y(test_indices, 1);
    Xtr = X(train_indices, :);
    Xte = X(test_indices, :);
toc
fprintf('Read data completed!\n');
end

