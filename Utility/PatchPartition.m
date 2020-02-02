function [patchVector patchNum]=PatchPartition(img,imgsize,patchsize)
imgNum=size(img,2);
BlockH=fix(imgsize(1)/patchsize(1));
BlockV=fix(imgsize(2)/patchsize(2));
patchNum=BlockH*BlockV;
patch_idx=[];
for i=1:BlockV
    for j=1:BlockH
        temp_patch = zeros(imgsize(1),imgsize(2));
        temp_patch((j-1)*patchsize(1)+1:j*patchsize(1), (i-1)*patchsize(2)+1:i*patchsize(2)) = 1;
        temp_idx = find(temp_patch==1);
        patch_idx = [patch_idx; temp_idx];
    end
end
patchVector=zeros(patchsize(1)*patchsize(2),patchNum,imgNum);
for i=1:imgNum
    temp_img=img(:,i);
    temp_img=temp_img(patch_idx);
    patchVector(:,:,i)=reshape(temp_img,patchsize(1)*patchsize(2),patchNum);
end
