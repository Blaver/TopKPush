function model = topKPushPreciseTrain(X, y, opt)
%TOP_K_PUSH The Top K Push Ranking Algorithm

%% load the options
if ~isfield(opt,'lambda')	opt.lambda=1;       end     % radius of l2 ball
if ~isfield(opt,'maxIter')	opt.maxIter=10000;	end     % maximal number of iterations
if ~isfield(opt,'tol')      opt.tol=1e-4;       end     % the relative gap
if ~isfield(opt,'debug')    opt.debug=false;    end     % the flag whether it is for debugging
if ~isfield(opt,'k')    opt.k=0.1;    end     %rate of k
if ~isfield(opt,'pj_mhd')    opt.pj_mhd = [];    end     %rate of k

debug = opt.debug;

lambda = opt.lambda;
maxIter=opt.maxIter;
tol=opt.tol;
k = opt.k;
delta=1e-6;

%% initialization
Xpos = X(y==1, :);      % positive instances
Xneg = X(y==-1, :);     % negative instances
m = sum(y==1);          % number of positive instances
n = sum(y==-1);         % number of negative instances

L = 1/m;
a=zeros(m,1);    ap=zeros(m,1);   aap=zeros(m,1);
q=zeros(n,1);    qp=zeros(n,1);   qqp=zeros(n,1);
t=1;    tp=0;   stop_flag=false;

if debug
%for testing , restore a and q
    A = zeros(m, 1);
    B = zeros(n, 1);
    pA = zeros(m, 1);
    pB = zeros(n, 1);
end

fun_val = zeros(maxIter, 1);

%% Nesterov's method (To slove a and q)
for iter = 1 : maxIter
    % --------------------------- step 1 ---------------------------
    % compute search point s based on ap (qp) and a (q) (with beta)
    beta=(tp-1)/t;    sa=a+beta* aap; sq=q+beta*qqp;
    
    % --------------------------- step 2 ---------------------------
    % line search for L and compute the new approximate solution x
    
    v = sa'*Xpos - sq'*Xneg;
    % compute the gradient and function value at s
    gd_a = Xpos*v'/(lambda*m)+sa/2-1;              % gradient on a
    gd_q = Xneg*v'/(-lambda*m);                    % gradient on q
    f_s = v*v'/(2*lambda*m)-sum(sa)+sa'*sa/4;     % function value
    
    % set ap=a  qp=q
    ap=a;   qp=q;
    
    while true
        % let sa walk in a step in the anti-gradient of sa to get va
        % and project va onto the line
        va = sa-gd_a/L;
        vq = sq-gd_q/L;
        
        %for testing, restore a and q
        if debug
            A = [A, va];
            B = [B, vq];
        end
        
        % euclidean projection onto the line (equality constraint)
        K = ceil(k * n);
        opt.step_sz = 2;
        opt.tol = 1e-5;
        opt.debug = 0;
        %[a, q] = preciseProject(va, vq, K, opt);
        %[a, q] = precise_project(va, vq, K);
        [a, q] = precise_project_improved(va, vq, K);
        
        if debug
            pA = [pA, a];
            pB = [pB, q];
        end
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % compute the objective value at the new approximate solution
        v = a'*Xpos-q'*Xneg;
        f_new = v*v'/(2*lambda*m)-sum(a)+a'*a/4;
        
        df_a = a-sa;    df_q = q-sq;
        r_sum=df_a'*df_a + df_q'*df_q;
        if ( sqrt(r_sum) <= delta)
            if debug
                fprintf('\n The distance between the searching point and the approximate is %e, and is smaller than %e \n',...
                    sqrt(r_sum), delta);
            end
            stop_flag=true;
            break;
        end
        
        l_sum = f_new - f_s - gd_a'*df_a - gd_q'*df_q;
        % the condition is l_sum <= L * r_sum
        if(l_sum <= r_sum*L*0.5)
            break;
        else
            L=2*L;
        end
    end
    
    % --------------------------- step 3 ---------------------------
    % update a and q, and check whether converge
    tp=t;   t=(1+sqrt(4*t*t+1))/2;
    aap=a-ap;    qqp=q-qp;
    
    fun_val(iter) = f_new/m;
    
    % ----------- check the stop condition
    if ( (iter >=10) && ...
            abs(fun_val(iter) - fun_val(iter-1)) <= abs(fun_val(iter-1))* tol)
        if debug
            fprintf('\n Relative obj. gap is less than %e \n',tol);
        end
        stop_flag=1;
    end
    
    if stop_flag
        break;
    end
%     if debug
%         fprintf('%d : %f  %d\n', iter, fun_val(iter), L);
%     end
end

%% Recover w using a and q
w = v'/(lambda*m);

%% Return result model
model.w = w;
if debug
    model.A = A;
    model.B = B;
    model.pA = pA;
    model.pB = pB;
end
model.name = 'TOPPUSH';
end




