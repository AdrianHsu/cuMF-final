/*
*	NMF by alternative non-negative least squares using projected gradients
*	Original Author: Chih-Jen Lin, National Taiwan University
*	Original website: http://www.csie.ntu.edu.tw/~cjlin/nmf/index.html
*	CUDA program auther: Dong Li (donggeat@gmail.com)
*	Created at 2014-09-16, modified at 2016-11-21 thank Zibin Zhang
*	I rewrote the Matlab code in CUDA, with CUBLAS. Free to use and modify.

*	As the first time to use CUDA, you may need a CUDA supported GPU at first.
*/

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <errno.h>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define bool int
#define false 0	
#define true 1
#define max(a,b)(a>b)?a:b
#define CUBLAS_ERROR_CHECK(sdata) if(CUBLAS_STATUS_SUCCESS!=sdata){printf("ERROR at:%s:%d\n",__FILE__,__LINE__);exit(-1);}
#define CUDA_ERROR_CHECK(sdata) if(cudaSuccess!=sdata){printf("cudaERROR at:%s:%d\n",__FILE__,__LINE__);exit(-1);}


void initializeCUDA(int &devID)
{
    //By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    devID = 1;

    // get number of SMs on this GPU
    CUDA_ERROR_CHECK(cudaGetDevice(&devID));

    cudaDeviceProp deviceProp;

    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&deviceProp, devID));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

}

void OutPutVector(int N, double* A)
{
	printf("Vector: %d\n",N);
	int i;	
	for(i=0;i<N;i++)
   		printf("%10.4f ",*(A+i));
	printf("\n");
}

void randomInit(double *data, int p)
{
	
    for (int i = 0; i < p; ++i)
	    data[i] = rand() / (double)RAND_MAX;
        //data[i] = 1.0;
}

//A:m*n,column-major; B=A'
void transpose(double *A, double *B, int m, int n){
	int k = 0;
	for( int i = 0; i < m; i++){
		for (int j = 0; j < n; j++)
			B[k++] = A[i+j*m];
	}		
} 

//W:m*n, H:n*k, V:m*k, grad:n*k
//At current, all the input matrices are in main memory, and all the out matrices should be in main memory as well.
void nlssubprob(double *V, double *W, double *Hinit, int m, int n, int k, double tol, int maxiter, double *H, double *grad, int *ite){
	//H = Hinit; WtV = W'*V; WtW = W'*W; 
	memcpy(H, Hinit, n*k*sizeof(double));
	
	cublasHandle_t handle;
        cublasStatus_t status;
	status = cublasCreate(&handle);CUBLAS_ERROR_CHECK(status);

	double alpha = 1;
	double beta = 0.1;

	double alpha1 = 1;
	double beta1 = 0;

	double* d_grad;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_grad, k*n*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemcpy(d_grad, grad, k*n*sizeof(double), cudaMemcpyHostToDevice));

	double* d_H;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_H, k*n*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemcpy(d_H, H, k*n*sizeof(double), cudaMemcpyHostToDevice));

	double* d_W;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_W, m*n*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemcpy(d_W, W, m*n*sizeof(double), cudaMemcpyHostToDevice));

	double* d_V;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_V, k*m*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemcpy(d_V, V, k*m*sizeof(double), cudaMemcpyHostToDevice));
	
	double *d_WtV = 0;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_WtV, n*k*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemset(d_WtV, 0, n*k*sizeof(double)));
	status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, k, m, &alpha1, d_W, m, d_V, m, &beta1, d_WtV, n);CUBLAS_ERROR_CHECK(status);
	
	double *d_WtW = 0;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_WtW, n*n*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemset(d_WtW, 0, n*n*sizeof(double)));

	status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &alpha1, d_W, m, d_W, m, &beta1, d_WtW, n);CUBLAS_ERROR_CHECK(status);
	
	double *Hn = 0;	
	Hn = (double *)malloc(n*k*sizeof(double));
	memset(Hn, 0, n*k*sizeof(double));
	
	double *d_Hn = 0;	
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_Hn, k*n*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemset(d_Hn, 0, k*n*sizeof(double)));

	double *d_d = 0;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_d, k*n*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemset(d_d, 0, k*n*sizeof(double)));
	
	double *d_WtWd = 0;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_WtWd, k*n*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemset(d_WtWd, 0, k*n*sizeof(double)));
	
	double *d_Hp = 0;	
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_Hp, k*n*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemset(d_Hp, 0, k*n*sizeof(double)));
	
	double *d_Hnpp = 0;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_Hnpp, k*n*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemset(d_Hnpp, 0, k*n*sizeof(double)));
	
	double *tmpvec = 0;		
	tmpvec = (double *)malloc(n*k*sizeof(double));
	memset(tmpvec, 0, n*k*sizeof(double));

	double *d_tmpvec = 0;		
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_tmpvec, k*n*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemset(d_tmpvec, 0, k*n*sizeof(double)));

	int iter = 0;
	for ( iter = 1; iter <= maxiter; iter++){
		//grad = WtW*H - WtV;
		CUDA_ERROR_CHECK(cudaMemcpy(d_grad, d_WtV, k*n*sizeof(double), cudaMemcpyDeviceToDevice));
		beta1 = -1;
		alpha1 = 1;
		status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &alpha1, d_WtW, n, d_H, n, &beta1, d_grad, n); CUBLAS_ERROR_CHECK(status);
		CUDA_ERROR_CHECK(cudaMemcpy(grad, d_grad, k*n*sizeof(double), cudaMemcpyDeviceToHost));

		//projgrad = norm(grad(grad < 0 | H >0));

		memset(tmpvec, 0, n*k*sizeof(double));
		int ii = 0;
		for (int i = 0; i < n*k; i++){
			if (grad[i] < 0 || H[i] > 0 )
				tmpvec[ii++] = grad[i];
		}
		CUDA_ERROR_CHECK(cudaMemcpy(d_tmpvec, tmpvec, ii*sizeof(double), cudaMemcpyHostToDevice));
		double projgrad = 0;
		status = cublasDnrm2(handle, ii, d_tmpvec, 1, &projgrad); CUBLAS_ERROR_CHECK(status);
		if (projgrad < tol)
			break;
		bool decr_alpha = true;
		for (int inner_iter = 1; inner_iter <= 20; inner_iter++){
			//Hn = max(H - alpha*grad, 0); d = Hn-H;
			CUDA_ERROR_CHECK(cudaMemcpy(d_Hn, d_H, n*k*sizeof(double), cudaMemcpyDeviceToDevice));
			alpha1 = -alpha;
			status = cublasDaxpy(handle, n*k, &alpha1, d_grad, 1, d_Hn, 1);CUBLAS_ERROR_CHECK(status);
			CUDA_ERROR_CHECK(cudaMemcpy(Hn, d_Hn, n*k*sizeof(double), cudaMemcpyDeviceToHost));
			for (int i = 0; i < n*k; i++){
				if (Hn[i] < 0)
					Hn[i] = 0;
			}
			CUDA_ERROR_CHECK(cudaMemcpy(d_Hn, Hn, n*k*sizeof(double), cudaMemcpyHostToDevice));
			CUDA_ERROR_CHECK(cudaMemcpy(d_d, d_Hn, n*k*sizeof(double), cudaMemcpyDeviceToDevice));
			alpha1 = -1;
			status = cublasDaxpy(handle, n*k, &alpha1, d_H, 1, d_d, 1);CUBLAS_ERROR_CHECK(status);
			
			//gradd=sum(sum(grad.*d)); dQd = sum(sum((WtW*d).*d));
			double gradd = 0;
			status = cublasDdot (handle, k*n, d_grad, 1, d_d, 1, &gradd);CUBLAS_ERROR_CHECK(status);
			alpha1 = 1;
			beta1 = 0;
			status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &alpha1, d_WtW, n, d_d, n, &beta1, d_WtWd, n);CUBLAS_ERROR_CHECK(status);

			double dQd = 0;
			status = cublasDdot (handle, k*n, d_WtWd, 1, d_d, 1, &dQd);CUBLAS_ERROR_CHECK(status);CUBLAS_ERROR_CHECK(status);

			bool suff_decr = 0.99*gradd + 0.5*dQd < 0;
			//bool decr_alpha = true;
			if (inner_iter==1){
				decr_alpha = ~suff_decr; 
				CUDA_ERROR_CHECK(cudaMemcpy(d_Hp, d_H, k*n*sizeof(double), cudaMemcpyDeviceToDevice));
			}
			if(decr_alpha){
				if(suff_decr){
					CUDA_ERROR_CHECK(cudaMemcpy(d_H, d_Hn, k*n*sizeof(double), cudaMemcpyDeviceToDevice));
					break;
				}
				else
					alpha = alpha * beta;
			}
			else{
				CUDA_ERROR_CHECK(cudaMemcpy(d_Hnpp, d_Hn, n*k*sizeof(double), cudaMemcpyDeviceToDevice));
				alpha1 = -1; 
				status = cublasDaxpy(handle, n*k, &alpha1, d_Hp, 1, d_Hnpp, 1);CUBLAS_ERROR_CHECK(status);
				double result  = 0;
				status = cublasDnrm2(handle, n*k, d_Hnpp, 1, &result);CUBLAS_ERROR_CHECK(status);
				if(~suff_decr || result == 0){
					CUDA_ERROR_CHECK(cudaMemcpy(d_H, d_Hp, k*n*sizeof(double), cudaMemcpyDeviceToDevice));
					break;
				}
				else{
					alpha = alpha/beta;
					CUDA_ERROR_CHECK(cudaMemcpy(d_Hp, Hn, k*n*sizeof(double), cudaMemcpyHostToDevice));
				}
			}
		}
		CUDA_ERROR_CHECK(cudaMemcpy(H, d_H, n*k*sizeof(double), cudaMemcpyDeviceToHost));
	}

	
	*ite = iter;
	if (*ite==maxiter)
		printf("Max iter in nlssubprob\n");

	free(Hn);
	free(tmpvec);

	CUDA_ERROR_CHECK(cudaFree(d_grad));
	CUDA_ERROR_CHECK(cudaFree(d_H));
	CUDA_ERROR_CHECK(cudaFree(d_W));
	CUDA_ERROR_CHECK(cudaFree(d_V));
	CUDA_ERROR_CHECK(cudaFree(d_WtV));
	CUDA_ERROR_CHECK(cudaFree(d_WtW));
	CUDA_ERROR_CHECK(cudaFree(d_Hn));
	CUDA_ERROR_CHECK(cudaFree(d_d));
	CUDA_ERROR_CHECK(cudaFree(d_WtWd));
	CUDA_ERROR_CHECK(cudaFree(d_Hp));
	CUDA_ERROR_CHECK(cudaFree(d_Hnpp));
	CUDA_ERROR_CHECK(cudaFree(d_tmpvec));

	status = cublasDestroy(handle);CUBLAS_ERROR_CHECK(status);
	
}

//NMF:V=W*H,W:m*n, H:n*k, V:m*k
//stick to BLAS, column-major
//At current, all the input matrices are in main memory, and all the out matrices should be in main memory as well.
//We need to transfer the input into device memory and the result out of it.
void NMF(double *V, double *Winit,double *Hinit, int m, int n, int k, double tol, double timelimit, int maxiter, double *W, double *H)
{
	
	memcpy(W,Winit,m*n*sizeof(double));
	memcpy(H,Hinit,n*k*sizeof(double));

        cublasHandle_t handle;
        cublasStatus_t status;
	status = cublasCreate(&handle);CUBLAS_ERROR_CHECK(status);
	
	double* d_V;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_V, k*m*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemcpy(d_V, V, k*m*sizeof(double), cudaMemcpyHostToDevice));

	//W = Winit; H = Hinit; initt = cputime;
	double* d_W;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_W, m*n*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemcpy(d_W, Winit, m*n*sizeof(double), cudaMemcpyHostToDevice));
	
	double* d_H;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_H, k*n*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemcpy(d_H, Hinit, k*n*sizeof(double), cudaMemcpyHostToDevice));
	

	clock_t initt = time(NULL);
	
	//gradW = W*(H*H') - V*H'; gradH = (W'*W)*H - W'*V;
	double *d_HHt = 0;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_HHt, n*n*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemset(d_HHt, 0, n*n*sizeof(double)));
	
	//cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n,n,k,1,H,n,H,n,0,HHt,n);
	double alpha = 1;
	double beta = 0;
	status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k, &alpha, d_H, n, d_H, n, &beta, d_HHt, n);CUBLAS_ERROR_CHECK(status);
	
	double* gradW;
	gradW = (double *)malloc(m*n*sizeof(double));
	memset(gradW, 0, m*n*sizeof(double));

	double * d_gradW = 0;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_gradW, m*n*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemset(d_gradW, 0, m*n*sizeof(double)));
	status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, &alpha, d_W, m, d_HHt, n, &beta, d_gradW, m);CUBLAS_ERROR_CHECK(status);
	
	double *d_VHt = 0;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_VHt, m*n*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemset(d_VHt, 0, m*n*sizeof(double)));
	status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, d_V, m, d_H, n, &beta, d_VHt, m);CUBLAS_ERROR_CHECK(status);
	alpha = -1;
	status = cublasDaxpy(handle, m*n, &alpha, d_VHt, 1, d_gradW, 1);CUBLAS_ERROR_CHECK(status);
	
	double *d_WtW = 0;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_WtW, n*n*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemset(d_WtW, 0, n*n*sizeof(double)));
	alpha = 1;
	status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &alpha, d_W, m, d_W, m, &beta, d_WtW, n);CUBLAS_ERROR_CHECK(status);

	double* gradH;
	gradH = (double *)malloc(k*n*sizeof(double));
	memset(gradH, 0, k*n*sizeof(double));

	double *d_gradH = 0;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_gradH, n*k*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemset(d_gradH, 0, n*k*sizeof(double)));
	status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &alpha, d_WtW, n, d_H, n, &beta, d_gradH, n);CUBLAS_ERROR_CHECK(status);
	
	double *d_WtV = 0;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_WtV, n*k*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemset(d_WtV, 0, n*k*sizeof(double)));
	status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, k, m, &alpha, d_W, m, d_V, m, &beta, d_WtV, n);CUBLAS_ERROR_CHECK(status);
	alpha = -1;
	status = cublasDaxpy(handle, n*k, &alpha, d_WtV, 1, d_gradH, 1);CUBLAS_ERROR_CHECK(status);
	
	double initgrad = 0, tempgad = 0;
	status = cublasDdot (handle, m*n, d_gradW, 1, d_gradW, 1, &initgrad);CUBLAS_ERROR_CHECK(status);
	status = cublasDdot (handle, m*n, d_gradH, 1, d_gradH, 1, &tempgad);CUBLAS_ERROR_CHECK(status);
	initgrad = sqrt(initgrad + tempgad);

	printf("Init gradient norm %f\n", initgrad); 
	double tolW = initgrad*max(0.001,tol); 
	double tolH = tolW;
	
	double *tmpvec = 0;
	tmpvec = (double *)malloc(m*n*sizeof(double));
	memset(tmpvec, 0, m*n*sizeof(double));
	
	double *tmpvec2 = 0;
	tmpvec2 = (double *)malloc(n*k*sizeof(double));
	memset(tmpvec2, 0, n*k*sizeof(double));

	double *d_tmpvec = 0;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_tmpvec, m*n*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemset(d_tmpvec, 0, m*n*sizeof(double)));
	
	double *d_tmpvec2 = 0;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_tmpvec2, k*n*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemset(d_tmpvec2, 0, n*k*sizeof(double)));
	
	double *Vt = 0;		
	Vt = (double *)malloc(m*k*sizeof(double));
	memset(Vt, 0, m*k*sizeof(double));
	transpose(V, Vt, k, m);

	double *Ht = 0;		
	Ht = (double *)malloc(n*k*sizeof(double));
	memset(Ht, 0, n*k*sizeof(double));
	
	double *Wt = 0;		
	Wt = (double *)malloc(m*n*sizeof(double));
	memset(Wt, 0, m*n*sizeof(double));
	

	CUDA_ERROR_CHECK(cudaMemcpy(gradW, d_gradW, m*n*sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_ERROR_CHECK(cudaMemcpy(gradH, d_gradH, k*n*sizeof(double), cudaMemcpyDeviceToHost));

	double projnorm = 0;
	double projnorm2 = 0;
	int iter = 0;	
	for (iter = 1; iter <= maxiter; iter++){
		//projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);
		
		int ii = 0;
		for (int i = 0; i < m*n; i++){
			if (gradW[i] < 0 || W[i] > 0 )
				tmpvec[ii++] = gradW[i];
		}
		CUDA_ERROR_CHECK(cudaMemcpy(d_tmpvec, tmpvec, ii*sizeof(double), cudaMemcpyHostToDevice));
		status = cublasDdot(handle, ii, d_tmpvec, 1, d_tmpvec, 1, &projnorm);CUBLAS_ERROR_CHECK(status);

		ii = 0;
		for (int i = 0; i < n*k; i++){
			if (gradH[i] < 0 || H[i] > 0 )
				tmpvec2[ii++] = gradH[i];
		}
		CUDA_ERROR_CHECK(cudaMemcpy(d_tmpvec2, tmpvec2, ii*sizeof(double), cudaMemcpyHostToDevice));
		status = cublasDdot (handle, ii, d_tmpvec2, 1, d_tmpvec2, 1, &projnorm2);CUBLAS_ERROR_CHECK(status);
		projnorm = sqrt(projnorm + projnorm2);
	    printf("\nIter = %d Final proj-grad norm %f\n", iter, projnorm);
	    printf("tol*initgrad = %f\n", tol*initgrad);
		if ( projnorm < tol*initgrad || time(NULL) - initt > timelimit)
			break;
		
		transpose(H, Ht, n, k);	
		transpose(W, Wt, m, n);

		//[W,gradW,iterW] = nlssubprob(V',H',W',tolW,1000);
		int iterW = 0;
		nlssubprob(Vt, Ht, Wt, k, n, m, tolW, 1000, W, gradW, &iterW);
	
		transpose(W, W, m, n);
		transpose(gradW, gradW, m, n);

		if (iterW == 1)
			tolW = 0.1 * tolW;
		
		int iterH = 0;
		nlssubprob(V, W, H, m, n, k, tolH, 1000, H, gradH, &iterH);
		if (iterH == 1)
			tolH = 0.1 * tolH;
		
		if ( iter%10 == 0)
			printf(".");
	}
	CUDA_ERROR_CHECK(cudaMemcpy(d_W, W, m*n*sizeof(double), cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaMemcpy(d_H, H, k*n*sizeof(double), cudaMemcpyHostToDevice));

	double *d_Vlast = 0;
	CUDA_ERROR_CHECK(cudaMalloc((void **)&d_Vlast, m*k*sizeof(double)));
	CUDA_ERROR_CHECK(cudaMemset(d_Vlast, 0, m*k*sizeof(double)));
	alpha = 1;
	beta = 0;		
	status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &alpha, d_W, m, d_H, n, &beta, d_Vlast, m);CUBLAS_ERROR_CHECK(status);
	
	CUDA_ERROR_CHECK(cudaMemcpy(V, d_Vlast, k*m*sizeof(double), cudaMemcpyDeviceToHost));
	printf("values in W*H (column major) are:\n");
	//OutPutVector(m*k, V);
	
	alpha = -1;
	status = cublasDaxpy(handle, m*k, &alpha, d_V, 1, d_Vlast, 1);CUBLAS_ERROR_CHECK(status);
	double fnorm = 0;
	status = cublasDdot (handle, m*k, d_Vlast, 1, d_Vlast, 1, &fnorm);CUBLAS_ERROR_CHECK(status);
	fnorm = sqrt(fnorm);
	printf("The ||W*H-V||_F is: %f\n", fnorm);
	
	free(gradW);
	free(gradH);
	free(tmpvec);
	free(tmpvec2);
	free(Vt);
	free(Wt);
	free(Ht);

	CUDA_ERROR_CHECK(cudaFree(d_W));
	CUDA_ERROR_CHECK(cudaFree(d_V));
	CUDA_ERROR_CHECK(cudaFree(d_H));
	CUDA_ERROR_CHECK(cudaFree(d_HHt));
	CUDA_ERROR_CHECK(cudaFree(d_gradW));
	CUDA_ERROR_CHECK(cudaFree(d_VHt));
	CUDA_ERROR_CHECK(cudaFree(d_WtW));
	CUDA_ERROR_CHECK(cudaFree(d_gradH));
	CUDA_ERROR_CHECK(cudaFree(d_WtV));
	CUDA_ERROR_CHECK(cudaFree(d_Vlast));

	status = cublasDestroy(handle);CUBLAS_ERROR_CHECK(status);
}

int main(int argc, char **argv)
{	

	int m = 943;//2;
	int n = 100;//3;
	int k = 1682;//4;

	int devID = 1;
    initializeCUDA(devID);
	
	srand((unsigned)time(NULL));

	//double V[8]={1,2,3,4,5,6,7,8};	
	double *V = 0;		
	V = (double *)malloc(m*k*sizeof(double));
    std::ifstream file("ml100k");
    std::string input; 
    while (std::getline(file, input))
    {
        std::istringstream ss(input);
        std::string token;
        int arr[3] = {0, 0, 0};
        for(int i = 0; i < 3; i++) {
            std::getline(ss, token, ' ');
            //std::cout << token << '\n';
            arr[i] = std::stoi(token);
        }
        //std::cout << arr[0] << ',' << arr[1] << '\n';
        V[ (arr[0] - 1) * k +  (arr[1] - 1) ] = ((double)arr[2]) * 1.0;
    }

	double *Winit = 0;		
	Winit = (double *)malloc(m*n*sizeof(double));

	double *W= 0;		
	W = (double *)malloc(m*n*sizeof(double));
	memset(W, 0, m*n*sizeof(double));

	double *Hinit = 0;		
	Hinit = (double *)malloc(n*k*sizeof(double));
	double *H= 0;		
	H = (double *)malloc(n*k*sizeof(double));
	memset(H, 0, n*k*sizeof(double));

	randomInit(Winit, m*n);
	randomInit(Hinit, n*k);

	NMF(V, Winit, Hinit, m, n, k, 1e-6, 1000, 1000, W, H);
	
	free(Hinit);
	free(Winit);
	free(H);
	free(W);

    CUDA_ERROR_CHECK(cudaDeviceReset());
	exit(1);
}
