gcc NMF.c  -o NMF.o -c -O3 -DADD_ -std=c99
gfortran -o NMF_ex NMF.o /tmp3/4dr14nh5u/nla/CBLAS/lib/cblas_LINUX.a /tmp3/4dr14nh5u/nla/CBLAS/BLAS-3.8.0/blas_LINUX.a
./NMF_ex
