function reco_rate=LUMIRCML_O_C(b,c0,lambda,noise_l)
dat_ad = [cd '/database/'];
load([dat_ad 'Subset1_DAT.mat']); 
load([dat_ad 'Subset2_DAT.mat']);
load([dat_ad 'Subset3_DAT.mat']);

D          = [subset1_data subset2_data];
D_labels   = [subset1_labels subset2_labels];
ll = size(D,2);
Test_DAT   = subset3_data;
testlabels = subset3_labels;
clear subset1_data subset2_data subset3_data

classids   = unique(D_labels);
classnum   = length(classids);

im_h       = 96;
im_w       = 84;
npix = im_h * im_w;

mean_x = 255*mean(D,2);
mu=2; 
nit        =  1;
nIter      =  25;  % for simple, we just do 10 iterations.
MEDIAN_A   =  [0.5];
block_l=0.5;
D          =  D./ repmat(sqrt(sum(D.*D)),[size(D,1) 1]);
median_a   = MEDIAN_A(1);
ID         =  [];

height     =   floor(sqrt(im_h*im_w*block_l));
width      =   height;
num_c      =   size(Test_DAT,2);
% width_rand = rand(num_c,1);
% height_rand = rand(num_c,1);
% save rand_w_h width_rand height_rand;
load rand_w_h;    % the random position created by the author

w_a = 1;w_b=im_w-width+1;
r_w = w_a + (w_b-w_a).*width_rand;
h_a = 1;h_b=im_h-height+1;
r_h = h_a + (h_b-h_a).*height_rand;
if height>=im_h||width>=im_w
    height=floor(im_h*sqrt(block_l));
    width=floor(im_w*sqrt(block_l));
end
RandS = RandStream('mt19937ar','Seed',1);
for pro_i = 1:size(Test_DAT,2)
    w_array=[];
    I    =   reshape(Test_DAT(:,pro_i),[96 84]);
    [J,ind_Occ]  =   Random_Block_Occlu(uint8(255.*I),r_h(pro_i),...
         r_w(pro_i),height,width);
     if noise_l>0
        [Q,ind_Corr] = Random_Pixel_Crop(uint8(J),noise_l,RandS);
         y = double(Q(:));
     else
         y =double(J(:));
     end

%      residual           =   (y-mean_x).^2;
     alpha = ones(ll,1)/ll;
     E = y-mean_x;
     residual = abs(E);
%      residual_sort = sort(residual);
%      iter = residual_sort(ceil(median_a*length(residual))); 
     iter = mean(residual);
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


%         curr_lambda = lambda*length(W_y);
        curr_lambda = lambda;
        A = W_D ./ curr_lambda;
        curr_mu = mu*length(W_y)/norm(W_y,1); %!
%!                  curr_mu = mu*norm(W_y,1)/norm(W_y);
        curr_lamalpha = curr_lambda*alpha;
        % [lamalpha, r, n, timeSteps] = SolvePALM_CBM(A, W_y, curr_mu);
        if nit <= 20
            maxIter = 4;%!
        else
            maxIter = 200;
        end
        [lamalpha, r, n, ni, timeSteps,beta_out] = SolvePALM_CBM(A,W_y,curr_mu,curr_lamalpha,maxIter,curr_beta(index_w));

        curr_beta(index_w) = beta_out;

        alpha = lamalpha/curr_lambda;
        E = norm_y_D.*(y-D*alpha);
        residual = abs(E);
%         residual_sort = sort(residual);
%         iter = residual_sort(ceil(median_a*length(residual))); 
        iter = mean(residual);
        w = 1./(1+c0*exp((residual-iter)/(iter/b)));
        w_array = [w_array w];
     end

    for class = 1:classnum
        index_class = (D_labels == class);
        z1 = W_y - W_D(:,index_class)*alpha(index_class) - r;
        gap1(class) = z1(:)'*z1(:);
    end
     % index = find(gap1==min(gap1));
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
        