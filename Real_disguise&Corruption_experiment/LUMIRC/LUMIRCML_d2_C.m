function reco_rate=LUMIRCML_d2_C(b,c0,lambda,noise_l,di)
tem_fd = cd;
par.d_fd          =   [cd '/database/'];
% addpath([cd '\utilities\']);

%seting parameter
par.nClass        =   100;
par.nSample       =   7;
par.ID            =   [];
par.nameDatabase  =   'AR_disguise';

%loading data: This is the second experiment of AR with disguise. 

load([par.d_fd 'AR_DR_DAT']);
Tr_DAT = []; trls = [];
for ci = 1:100
    Tr_DAT = [Tr_DAT Tr_dataMatrix(:,1+7*(ci-1)) Tr_dataMatrix(:,5+7*(ci-1):7+7*(ci-1))];
    trls   = [trls repmat(ci,[1 4])];
end
clear Tr_dataMatrix Tr_sampleLabels Tt_dataMatrix Tt_sampleLabels;

load([par.d_fd 'AR_database_Occlusion.mat']);
Tt_DAT_sunglass1 = [];
ttls_sunglass1 = [];
Tt_DAT_scarf1 = []; 
ttls_scarf1 = [];
Tt_DAT_sunglass2 = [];
ttls_sunglass2 = [];
Tt_DAT_scarf2 = []; 
ttls_scarf2 = [];
for ci = 1:100
    Tt_DAT_sunglass1 = [Tt_DAT_sunglass1 Tr_dataMatrix(:,1+6*(ci-1):3+6*(ci-1))]; % Session 1

    %     session 2
    Tt_DAT_sunglass2 = [Tt_DAT_sunglass2 Tt_dataMatrix(:,1+6*(ci-1):3+6*(ci-1))]; %
    ttls_sunglass1   = [ttls_sunglass1 repmat(ci,[1 3])];
    ttls_sunglass2   = [ttls_sunglass2 repmat(ci,[1 3])];
    Tt_DAT_scarf1 = [Tt_DAT_scarf1 Tr_dataMatrix(:,4+6*(ci-1):6+6*(ci-1))]; % Session 1

    %     session 2
    Tt_DAT_scarf2 = [Tt_DAT_scarf2 Tt_dataMatrix(:,4+6*(ci-1):6+6*(ci-1))];
    ttls_scarf1   = [ttls_scarf1 repmat(ci,[1 3])];
    ttls_scarf2   = [ttls_scarf2 repmat(ci,[1 3])];
end
clear Tr_dataMatrix Tr_sampleLabels Tt_dataMatrix Tt_sampleLabels;
nRow = 42;
nCol = 30;
for i = 1:size(Tr_DAT,2)
    tem = reshape(Tr_DAT(:,i),[165 120]);
    tem1 = uint8(imresize(tem,[nRow nCol]));
    O_Tr_DAT(:,i) = tem1(:);
end
O_Tr_DAT = double(O_Tr_DAT);
mean_x = mean(O_Tr_DAT,2);  % 均值在0-255之间，该步骤很重要
ll = size(O_Tr_DAT,2);

if di==1
    Tt_DAT              =  Tt_DAT_sunglass1;
    ttls                =  ttls_sunglass1;
elseif di==2
    Tt_DAT              =  Tt_DAT_sunglass2;
    ttls                =  ttls_sunglass2;
elseif di==3
    Tt_DAT              =  Tt_DAT_scarf1;
    ttls                =  ttls_scarf1;
else
    Tt_DAT              =  Tt_DAT_scarf2;
    ttls                =  ttls_scarf2;
end

for i = 1:size(Tt_DAT,2)
    tem = reshape(Tt_DAT(:,i),[165 120]);
    tem1 = uint8(imresize(tem,[nRow nCol]));
    O_Tt_DAT(:,i) = tem1(:);
end


O_Tt_DAT = double(O_Tt_DAT);

D         = O_Tr_DAT;
D_labels  = trls;
Test_DAT  = O_Tt_DAT;
testlabels= ttls;

classids   = unique(D_labels);
classnum   = length(classids);

npix = nRow * nCol;  

nit        =  1;
nIter      =  25;  
median_a=0.5;
D          =  D./ repmat(sqrt(sum(D.*D)),[size(D,1) 1]); 
mu=2;
RandS = RandStream('mt19937ar','Seed',1);
ID         =  [];
for pro_i = 1:size(Test_DAT,2)

    I = reshape(Test_DAT(:,pro_i),[nRow nCol]);
    if noise_l>0
        [J,ind_R] = Random_Pixel_Crop(uint8(I),noise_l,RandS);
        y = double(J(:));
    else
        y=double(I(:));
    end
    alpha = ones(ll,1)/ll;
    E = y-mean_x;
    residual = abs(E);
%     residual_sort = sort(residual);
%     iter = residual_sort(ceil(median_a*length(residual))); %! beta = beta_a/iter; 
    iter = mean(residual);%!
    w = 1./(1+c0*exp((residual-iter)/(iter/b)));
    norm_y_D = norm(y);
    y = y./norm(y);

    curr_beta = zeros(length(y),1);
    for nit = 1: nIter

        tem_w = w./max(w);
        index_w = find(tem_w>1e-3);
        % remove the pixels with very small weight
        W_y = w(index_w).*y(index_w);
        W_D = repmat(w(index_w),[1 size(D,2)]).*D(index_w,:);

%         curr_lambda = lambda*length(W_y);%!
        curr_lambda = lambda;%!
        A = W_D ./ curr_lambda;
        curr_mu = mu*length(W_y)/norm(W_y,1); %!
        %! curr_mu = mu*norm(W_y,1)/norm(W_y);
        curr_lamalpha = curr_lambda*alpha;
        % [lamalpha, r, n, timeSteps] = SolvePALM_CBM(A, W_y, curr_mu);
        if nit <= 20
            maxIter = 4; %!
        else
            maxIter = 200;
        end
        [lamalpha, r, n, ni, timeSteps,beta_out] = SolvePALM_CBM(A,W_y,curr_mu,curr_lamalpha,maxIter,curr_beta(index_w));

        curr_beta(index_w) = beta_out;

        alpha = lamalpha/curr_lambda;
        E = norm_y_D.*(y-D*alpha);
        residual = abs(E);
%         residual_sort = sort(residual);
%         iter = residual_sort(ceil(median_a*length(residual))); %! beta = beta_a/iter; 
        iter = mean(residual);%!
        w = 1./(1+c0*exp((residual-iter)/(iter/b)));
%         figure;
%          imshow(reshape(w,nRow,nCol),[0 1])
    end


     for class = 1:classnum
        index_class = (D_labels == class);
        z1 = W_y - W_D(:,index_class)*alpha(index_class) - r;
        gap1(class) = z1(:)'*z1(:);
    end


    [gap1,index] = sort(gap1);
    id = index(1:10);
    gap1 = gap1(1:10);
    ratio_gap(pro_i) = gap1(2)/gap1(1);


    if id(1)==testlabels(pro_i)
        fprintf('.');
    else
        fprintf('*');
    end
    if ~mod(pro_i,100)
        fprintf('\n');
    end

    ID = [ID id(1)];
end

reco_rate = sum(ID == testlabels)/length(testlabels);
