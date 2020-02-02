function reco_rate=LUMIRCMLP_d2_C(b,c0,lambda,noise_l,di,patchsize)
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

% mean_x = mean(O_Tr_DAT,2);
ll = size(O_Tr_DAT,2);
D          = O_Tr_DAT ;
D_labels   = trls;
classids   = unique(D_labels);
classnum   = length(classids);
npix = nRow * nCol;  %ÿһ֡

imgsize=[nRow,nCol];
[D_Partition patchNum]=PatchPartition(D, imgsize,patchsize);
mean_x_Partition=mean(D_Partition,3);

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
Test_DAT   = O_Tt_DAT;
testlabels = ttls;
RandS = RandStream('mt19937ar','Seed',1);
ID         =  [];

nit        =  1;
nIter      =  25;  
median_a=0.5; 
for i=1:patchNum
    tempD=D_Partition(:,i,:);
    D_Partition(:,i,:)         =  tempD./ repmat(sqrt(sum(tempD.*tempD)),[size(tempD,1) 1]);
end
D          =  D./ repmat(sqrt(sum(D.*D)),[size(D,1) 1]); 
mu=2;
RandS = RandStream('mt19937ar','Seed',1);
ID         =  [];
for pro_i = 1:size(Test_DAT,2)
    idSet=[];
    I = reshape(Test_DAT(:,pro_i),[nRow nCol]);
    if noise_l>0
        [J,ind_R] = Random_Pixel_Crop(uint8(I),noise_l,RandS);
        y = double(J(:));
    else
        y=double(I(:));
    end
    y_Partition=PatchPartition(y, imgsize,patchsize);
    for patchidx=1:patchNum
        alpha = ones(ll,1)/ll;
        E_Partition = y_Partition(:,patchidx)-mean_x_Partition(:,patchidx);
        residual = abs(E_Partition);
    %     residual_sort = sort(residual);
    %     iter = residual_sort(ceil(median_a*length(residual))); %! beta = beta_a/iter; 
        iter = mean(residual); %!
        w = 1./(1+c0*exp((residual-iter)/(iter/b)));
        norm_y_D = norm(y_Partition(:,patchidx));
        y_Partition(:,patchidx) = y_Partition(:,patchidx)./norm(y_Partition(:,patchidx));

        curr_beta = zeros(length(y_Partition(:,patchidx)),1);
        for nit = 1: nIter

            tem_w = w./max(w);
            index_w = find(tem_w>1e-3);
            % remove the pixels with very small weight
            y_P=y_Partition(:,patchidx);
            W_y = w(index_w).*y_P(index_w);
            W_D = repmat(w(index_w),[1 size(D_Partition,3)]).*squeeze(D_Partition(index_w,patchidx,:));


    %         curr_lambda = lambda*length(W_y);%!
            curr_lambda = lambda;
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
            E_Partition = norm_y_D.*(y_Partition(:,patchidx)-squeeze(D_Partition(:,patchidx,:))*alpha);
            residual = abs(E_Partition);
    %         residual_sort = sort(residual);
    %         iter = residual_sort(ceil(median_a*length(residual))); %! beta = beta_a/iter; 
            iter = mean(residual); %!
            w = 1./(1+c0*exp((residual-iter)/(iter/b)));
        end


        for class = 1:classnum
            index_class = (D_labels == class);
            delta_x(class)=sum(abs(alpha(index_class)));
            z1 = W_y - W_D(:,index_class)*alpha(index_class) - r;
            gap1(class) = z1(:)'*z1(:);
        end
        SCI(patchidx)=(classnum*max(delta_x)/sum(abs(alpha))-1)/(classnum-1);
     % index = find(gap1==min(gap1));
        [gap1,index] = sort(gap1);
        id = index(1:10);
        gap1 = gap1(1:10);
        ratio_gap(pro_i) = gap1(2)/gap1(1);
        idSet=[idSet id(1)];
    end
    id_candidate=unique(idSet);
    score_candidate=[];
    for ii=1:length(id_candidate)
        score_candidate(ii)=sum((idSet==id_candidate(ii)).*SCI);
    end
    [score index ]=max(score_candidate);
    id_most=id_candidate(index);

    if id_most==testlabels(pro_i)
        fprintf('.');
    else
        fprintf('*');
    end
    if ~mod(pro_i,100)
        fprintf('\n');
    end

    ID = [ID id_most];
end

reco_rate = sum(ID == testlabels)/length(testlabels);
end

