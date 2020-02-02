function [J ,ind_R] = Random_Block_Occlu(I,r_h,r_w,height,width)
J = I;
[h,w] = size(I);
baroon = rgb2gray(imread('baboon.tif'));
J(round(r_h):round(r_h)+height-1,round(r_w):round(r_w)+width-1)= imresize(baroon,[height width]);
ind_R = false(h,w);
ind_R(round(r_h):round(r_h)+height-1,round(r_w):round(r_w)+width-1)=1;
ind_R=ind_R(:);

