# CuMF_SGD

## Introduction

Matrix factorization has been demonstrated to be effective in recommender system, topic modeling, word embedding, and other machine learning applications. As the input data set is often large, MF solution are time-consuming. Therefore, how to solve MF problems efficiently is an important problem. There are mainly three algorithms to solve MF, coordinate gradient descent(CGD), alternate least square(ALS), and stochastic gradient descent(SGD). Our previous project tackles [ALS](https://github.com/wei-tan/cumf_als) acceleration on GPUs, we foucs on SGD solution in this project and present **cuMF_SGD**.


<img src=https://github.com/CuMF/cumf_sgd/raw/master/figures/mf.png width=405 height=161 />


CuMF_SGD is a CUDA-based SGD solution for large-scale matrix factorization(MF) problems. CuMF_SGD is able to solve MF problems with one or multiple GPUs within one single node. It first partitions the input data into matrix blocks and distribute them to different GPUs. Then it uses batch-Hogwild! algorithm to parallelize the SGD updates withn one GPU. It also has highly-optimized kernels for SGD update, leveraging cache, warp-shuffle instructions, and half-precision floats.



We test cuMF_SGD using three data sets (Netflix, Yahoo!Music and Hugewiki) with one Maxwell or Pascal GPU, cumf_sgd runs 3.1X-28.2X as fast compared with state-of-art CPU solutions on 1-64 CPU nodes. We also test Yahoo!Music on two Pascal GPUs and we observer that two GPUs provides ~30% speedup over one GPU. 

Note: the repository only contains single GPU version, the multiple GPU version still needs more test.

## Compilation 

Run the Makefile in the source code directory.

## Input data format

The input rating is organized as follows:

    user_id item_id rating

user_id and item_id are 4-byte integers and rating is 4-byte floating point. They are all stored in binary format. 

Please download netflix data set here（[Netflix Data set](https://drive.google.com/drive/folders/1ZxG4hVWqNGnlvPwx0T7lDwDq816GLXv-?usp=sharing). Put all children files in ./data/netflix and move to next section.

The netflix_mm and netflix_mme are original data files downloaded from [netflix_mm](http://www.select.cs.cmu.edu/code/graphlab/datasets/netflix_mm) and [netflix_mme](http://www.select.cs.cmu.edu/code/graphlab/datasets/netflix_mme). As the download link no longer works, we put them on the above google drive link. If you are interested on how to transform "netflix_mm/netflix_mme" to "netflix_mm.bin/netflix_mme.bin", please check out ./data/netflix/prepare.sh


## Run
usage: 
    
    ./singleGPU/cumf_sgd [options] train_file [model_file]

options:<br />
-g <gpu_id>: specify the device id of the GPU(optional).<br />
-l <lambda>: l2 regularization parameter for both P and Q.<br />
-k <dimensions>: length of the factorization factor. Now cuMF_SGD only supports k = 128.<br />
-t <iterations>: number of iterations.<br />
-a <alpha>: initial learning rate.<br />
-b <beta>: learning rate scheduling parameter(see the paper for learning rate scheduling algorithm).<br />
-s <parallel workers>: number of parallel workers.<br />
-u :<br />
-v : first level partion parameters. We partition the input matrix into u * v blocks. Default is 1 * 1.<br />
-x :<br />
-y : For each partition, we further partion it into x * y blocks and overlap x * y blocks to minimize the memory transfer overhead. Default is 1*1.<br />


We have a run script for Netflix data set:

    ./data/netflix/run.sh

In this script, we set u, v, x, and y as 1 as the data set is enough to fit into one GPU. 

Developers can set parallel workers as 52 * NumberofSMs. On TITAN X GPU, the number is 24 * 52=1248. It works well. We also recommend you to fine tune this metric to maximize the performance. 


## Test
We use the same model file with [Libmf](https://github.com/cjlin1/libmf) but with different input file format. We adopt the test code of [Libmf](https://github.com/cjlin1/libmf) and modify it to adapt to our file format. You can run the following commands to run test for Netflix data set.
  
    cd test
    make
    cd ..
    ./data/netflix/test.sh


## Reference


Details can be found at:

Xiaolong Xie, [Wei Tan](https://github.com/wei-tan), [Liana Fong](https://github.com/llfong), Yun Liang, CuMF_SGD: Parallelized Stochastic Gradient Descent for Matrix Factorization on GPUs, ([arxiv link](https://arxiv.org/abs/1610.05838)).

Our ALS-based MF solution can be found here:

Faster and Cheaper: Parallelizing Large-Scale Matrix Factorization on GPUs. [Wei Tan](https://github.com/wei-tan), [Liangliang Cao](https://github.com/llcao), [Liana Fong](https://github.com/llfong). [HPDC 2016], Kyoto, Japan. [(arXiv)](http://arxiv.org/abs/1603.03820) [(github)](https://github.com/wei-tan/cumf_als)

