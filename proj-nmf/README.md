# Non-negative Matrix Factorization (CUDA)
## Course
Numerical Linear Algebra, 2017 Fall  
## Instructor
王偉仲 教授（WEICHUNG WANG） 
## Members
R05246005 徐唯恩 B03901023 許秉鈞 B03901041 王文謙 R06246001 陳博允

## PART 1. PROJGRAD NMF (CPU)


NMF(Non-negative Matrix Factorization) based on CBLAS, with dense matrix as input.

**NMF.c** This code solves NMF by alternative non-negative least squares using projected gradients. It's an implementation of [Projected gradient methods for non-negative matrix factorization](https://www.csie.ntu.edu.tw/~cjlin/papers/pgradnmf.pdf). You may read the paper for more details.

### How-to
As the first time to use BLAS and CBLAS, you may need to configure like this on Linux:
Download blas.tgz and cblas.tgz on http://www.netlib.org/blas/
1) Install BLAS, generate blas_LINUX.a
2) Modify the BLLIB in CBLAS/Makefile.in which link to blas_LINUX.a, and make all in CBLAS
3) Put the src/cblas.h to /usr/lib/ or somewhere your compiler can find it, then enjoy it!

### Run
```
compile:  gcc NMF.c  -o NMF.o -c -O3 -DADD_ -std=c99
     gfortran -o NMF_ex NMF.o /home/lid/Downloads/CBLAS/lib/cblas_LINUX.a /home/lid/Downloads/BLAS/blas_LINUX.a
execute:   ./NMF_ex
```

### Contribution 

In order to compare the GPU version algorithm with the normal one, we directly fork the CPU version provided by Professor Chih-Jen Lin's Website, [https://www.csie.ntu.edu.tw/~cjlin/nmf/](https://www.csie.ntu.edu.tw/~cjlin/nmf/), and the original implementation was done by **Dong Li**, you could download the piece of code from [https://www.csie.ntu.edu.tw/~cjlin/nmf/others/NMF.c](https://www.csie.ntu.edu.tw/~cjlin/nmf/others/NMF.c) 

## PART 2. PROJGRAD NMF (GPU Tesla K80)

NMF(Non-negative Matrix Factorization) based on cuda, with sparse matrix as input.

**NMF_pgd.cu** This code solves NMF by alternative non-negative least squares using projected gradients. It's an implementation of [Projected gradient methods for non-negative matrix factorization](https://www.csie.ntu.edu.tw/~cjlin/papers/pgradnmf.pdf). You may read the paper for more details.

### Requirements
The code is base on cuda, cuBlas and cuSparse precisely. Please get cuda from Nvidia's website, https://developer.nvidia.com/cuda-downloads.

### Usage
Results will be saved in two files, W.txt and H.txt in dense format.
You should use nvcc to compile the code, so make sure cuda is installed and environment is correctly setted.

```bash
$ make
$ ./NMF_pgd
```

The default data input file is ```Movie Lens 100K dataset```, you can download it from [https://github.com/AdrianHsu/cuMF-final/blob/master/proj-nmf/gpu/ml100k](https://github.com/AdrianHsu/cuMF-final/blob/master/proj-nmf/gpu/ml100k)

### Contribution  
The original work is done by ```zhangzibin: cu-nmf```, and we fixed some bugs, and then modified the input format for adapting the other datasets, you could access the original work from [https://github.com/zhangzibin/cu-nmf](https://github.com/zhangzibin/cu-nmf)


## PART 3. PROJGRAD NMF (python, matlab)
Since that these two implementations are not our main concern (we use them to compare and debug the C++ version), so if you have time to review it, you're welcomed to run our code here: [https://github.com/AdrianHsu/cuMF-final/tree/master/proj-nmf](https://github.com/AdrianHsu/cuMF-final/tree/master/proj-nmf)

