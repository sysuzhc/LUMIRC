#  Laplacian-Uniform Mixture-Driven Iterative Robust Coding With Applications to Face Recognition Against Dense Errors

### Introduction

This is the original **MATLAB** implementation for our TNNLS paper, ***Laplacian-Uniform Mixture-Driven Iterative Robust Coding With Applications to Face Recognition Against Dense Errors***. It reports a recognition accuracy of ***93.8%*** on EYB when 90% of the facial pixels are corrupted.

The MATLAB source codes for EYB experiments and AR experiments can be found in Occlusion&Corruption_experiment folder and Real_disguise&Corruption_experiment folder, corresponding to SectionIV-B and SectionIV-C in our paper, respectively. The corresponding datasets can be publicly accessed via Internet. The following is an overview of the proposed approach. Details can be found in our paper.

![image](https://github.com/sysuzhc/LUMIRC/blob/master/Idea.jpg)

The above figure illustrates the main idea of cooperative error detection and correction for robust face recognition. The weight image *W* detects errors in the input image **y**, so that sparse representation can be performed on reliable data. The errors which are not fully detected in challenging situations are further corrected to enforce a faithful reconstruction of the face image as a sparse linear combination of all the training images from dictionary *A* with coefficients **x** (red ones correspond to training images with the same class label as the input image). The cooperation of error detection and error correction leads to an extremely robust algorithm, which accurately identifies the subject from 400 training images of 100 individuals in the AR face database, even under very challenging scarf occlusion and pixel corruption.

###  Reference  

If you find it interesting and want to use it for discussion or comparison in your paper, please cite the following paper.

Huicheng Zheng, Dajun Lin, Lina Lian, Jiayu Dong, and Peipei Zhang, “Laplacian-Uniform mixture-driven iterative robust coding with applications to face recognition against dense errors,” ***IEEE Transactions on Neural Networks and Learning Systems***, 2019, early access, https://doi.org/10.1109/TNNLS.2019.2945372.

###  Contact

If you have any inquiry about the paper, please feel free to contact the corresponding author through email: zhenghch@mail.sysu.edu.cn
