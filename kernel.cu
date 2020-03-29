/* Code by Mohamed Kasem 
compile using this command:
nvcc kernel.cu -o mykernel -gencode arch=compute_35,code=sm_35 -lcublas
 This will generate code for any gpu with compute capability 3.5 and will link the cublas library 
 To know your Nvidia GPU compute capabilities, visit this website: 
 https://developer.nvidia.com/cuda-gpus
 This code has been compiled and tested on Nvidia GPU GT-920M and an x86 Intel CPU.
 Solves for [x1,x2,x3, ...] in [A1*x1 = y1, A2*x2 = y2, A3*x3 = y3, ...] . This code solves all of the equations in parallel
 To specify the batch size change mybatch variable in BackSlashOp() function below to implement piecewise modelling in a 
 least squares system.
*/

#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <cstdlib>
#include <time.h>
#include <cuda.h>

#define cudacall(call)                                                                                                        \
	do                                                                                                                        \
	{                                                                                                                         \
		cudaError_t err = (call);                                                                                             \
		if (cudaSuccess != err)                                                                                               \
		{                                                                                                                     \
			fprintf(stderr, "CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
			cudaDeviceReset();                                                                                                \
			exit(EXIT_FAILURE);                                                                                               \
		}                                                                                                                     \
	} while (0)

#define cublascall(call)                                                                                     \
	do                                                                                                       \
	{                                                                                                        \
		cublasStatus_t status = (call);                                                                      \
		if (CUBLAS_STATUS_SUCCESS != status)                                                                 \
		{                                                                                                    \
			fprintf(stderr, "CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status); \
			cudaDeviceReset();                                                                               \
			exit(EXIT_FAILURE);                                                                              \
		}                                                                                                    \
                                                                                                             \
	} while (0)

void invert(float **src, float **dst, int n, int batchSize, float &timerGPU, float &timerTotal)
{
	cudaEvent_t start, stop;
	cudaEvent_t st, sp;
	float t1 = 0;
	float timer_1 = 0;
	float timer_2 = 0;

	cudaEventCreate(&st);
	cudaEventCreate(&sp);
	cudaEventRecord(st, 0);

	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));

	int *P, *INFO;

	cudacall(cudaMalloc(&P, n * batchSize * sizeof(int)));
	cudacall(cudaMalloc(&INFO, batchSize * sizeof(int)));

	int lda = n;

	float **A = new float *[batchSize];
	float **A_d;
	float *A_dflat;

	cudacall(cudaMalloc(&A_d, batchSize * sizeof(float *)));
	cudacall(cudaMalloc(&A_dflat, n * n * batchSize * sizeof(float)));

	A[0] = A_dflat;
	for (int i = 1; i < batchSize; i++)
		A[i] = A[i - 1] + (n * n);
	cudacall(cudaMemcpy(A_d, A, batchSize * sizeof(float *), cudaMemcpyHostToDevice));

	for (int i = 0; i < batchSize; i++)
		cudacall(cudaMemcpy(A_dflat + (i * n * n), src[i], n * n * sizeof(float), cudaMemcpyHostToDevice));

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cublascall(cublasSgetrfBatched(handle, n, A_d, lda, P, INFO, batchSize));

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timer_1, start, stop);

	int *INFOh = new int[batchSize];
	cudacall(cudaMemcpy(INFOh, INFO, batchSize * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < batchSize; i++)
		if (INFOh[i] != 0)
		{
			fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}

	float **C = new float *[batchSize];
	float **C_d, *C_dflat;
	cudacall(cudaMalloc(&C_d, batchSize * sizeof(float *)));
	cudacall(cudaMalloc(&C_dflat, n * n * batchSize * sizeof(float)));
	C[0] = C_dflat;
	for (int i = 1; i < batchSize; i++)
		C[i] = C[i - 1] + (n * n);
	cudacall(cudaMemcpy(C_d, C, batchSize * sizeof(float *), cudaMemcpyHostToDevice));

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cublascall(cublasSgetriBatched(handle, n, (const float **)A_d, lda, P, C_d, lda, INFO, batchSize));

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timer_1, start, stop);

	timerGPU = timer_1 + timer_2;

	cudacall(cudaMemcpy(INFOh, INFO, batchSize * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < batchSize; i++)
		if (INFOh[i] != 0)
		{
			fprintf(stderr, "Inversion of matrix %d Failed: Matrix may be singular\n", i);
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}
	for (int i = 0; i < batchSize; i++)
		cudacall(cudaMemcpy(dst[i], C_dflat + (i * n * n), n * n * sizeof(float), cudaMemcpyDeviceToHost));

	delete[] INFOh;
	/*	for (int i = 0; i < batchSize; ++i)
		{delete [] A[i]; delete [] C[i];}*/
	delete[] A;
	delete[] C;

	cudaFree(A_d);
	cudaFree(A_dflat);
	cudaFree(C_d);
	cudaFree(C_dflat);
	cudaFree(P);
	cudaFree(INFO);
	cublasDestroy_v2(handle);

	cudaEventRecord(sp, 0);
	cudaEventSynchronize(sp);
	cudaEventElapsedTime(&timerTotal, st, sp);
}
void printArrayBatch(float **A, int rowA, int colA, int batchsize, char c = 'A')
{
	for (int q = 0; q < batchsize; q++)
	{
		fprintf(stdout, "%c", c);
		fprintf(stdout, "%d:\n\n", q);
		for (int i = 0; i < rowA; i++)
		{
			for (int j = 0; j < colA; j++)
				fprintf(stdout, "%f\t", A[q][i * colA + j]);
			fprintf(stdout, "\n");
		}
	}
}
void test_invert()
{
	const int n = 3;
	const int mybatch = 2;
	float timerGPU = -1;
	float timerTotal = -1;
	float **inputs = new float *[mybatch];

	for (int i1 = 0; i1 < mybatch; i1++)
		inputs[i1] = new float[n * n];

	for (int j = 0; j < mybatch; j++)
		for (int i = 0; i < n * n; i++)
			inputs[j][i] = (double)rand() / RAND_MAX;

	float **results = new float *[mybatch];
	for (int i = 0; i < mybatch; i++)
		results[i] = new float[n * n];

	if (n * n * mybatch < 10000)
	{
		for (int qq = 0; qq < mybatch; qq++)
		{
			fprintf(stdout, "Input %d:\n\n", qq);
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < n; j++)
					fprintf(stdout, "%f\t", inputs[qq][i * n + j]);
				fprintf(stdout, "\n");
			}
		}
		fprintf(stdout, "\n\n");
	}
	invert(inputs, results, n, mybatch, timerGPU, timerTotal);

	if (n * n * mybatch < 10000)
	{
		for (int qq = 0; qq < mybatch; qq++)
		{
			fprintf(stdout, "Inverse %d:\n\n", qq);
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < n; j++)
					fprintf(stdout, "%f\t", results[qq][i * n + j]);
				fprintf(stdout, "\n");
			}
		}
		/*	for (int i = 0; i < mybatch; ++i)
		delete [] results[i];

	for (int i = 0; i < mybatch; ++i)
		delete [] inputs[i];*/
		delete[] inputs;

		delete[] results;
	}

	fprintf(stdout, "Computation only time for GPU: %f ms\n", timerGPU);
	fprintf(stdout, "Total time for GPU: %f ms\n", timerTotal);
}

float Transpose(float *src, float *dst, int m, int n)
{
	float const alpha(1.0);
	float const beta(0.0);
	float *A_dflat;
	float *C_dflat;
	float timer;

	cublasHandle_t handle;
	cudaEvent_t start, stop;

	cublasCreate(&handle);

	cudacall(cudaMalloc(&A_dflat, m * n * sizeof(float *)));
	cudacall(cudaMalloc(&C_dflat, m * n * sizeof(float *)));
	cudacall(cudaMemcpy(A_dflat, src, m * n * sizeof(float), cudaMemcpyHostToDevice));

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cublascall(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, (float *)A_dflat, n, &beta, (float *)A_dflat, m, (float *)C_dflat, m));

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timer, start, stop);

	cudacall(cudaMemcpy(dst, C_dflat, m * n * sizeof(float), cudaMemcpyDeviceToHost));

	cublasDestroy(handle);

	cudaFree(A_dflat);
	cudaFree(C_dflat);

	return timer;
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)

void MatMul(float *A, float *B, float *C, const int rowA, const int colB, const int com)
{

	int lda = colB, ldb = com, ldc = colB;
	float alpha = 1.0;
	float beta = 0.0;

	float *A_d;
	float *B_d;
	float *C_d;
	float timer;
	cudaEvent_t start, stop;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	cudacall(cudaMalloc(&A_d, rowA * com * sizeof(float *)));
	cudacall(cudaMalloc(&B_d, com * colB * sizeof(float *)));
	cudacall(cudaMalloc(&C_d, rowA * colB * sizeof(float *)));

	cudacall(cudaMemcpy(A_d, A, rowA * com * sizeof(float), cudaMemcpyHostToDevice));
	cudacall(cudaMemcpy(B_d, B, com * colB * sizeof(float), cudaMemcpyHostToDevice));

	// Do the actual multiplication
	cublascall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, colB, rowA, com, &alpha, B_d, lda, A_d, ldb, &beta, C_d, ldc));

	cudacall(cudaMemcpy(C, C_d, rowA * colB * sizeof(float), cudaMemcpyDeviceToHost));

	// Destroy the handle
	cublasDestroy(handle);

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
}

float MatMulBatched(float **src1, float **src2, float **dist, const int rowA, const int colB, const int com, int batchSize)
{
	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	//Required for timing the gemm function
	cudaEvent_t start, stop;
	float timer_1 = 0;

	//leading dimensions
	int lda = colB, ldb = com, ldc = colB;
	float alpha = 1.0;
	float beta = 0.0;
	float **A = new float *[batchSize];
	float **B = new float *[batchSize];
	float **A_d, **B_d;
	float *A_dflat, *B_dflat;

	cudacall(cudaMalloc(&A_d, batchSize * sizeof(float *)));
	cudacall(cudaMalloc(&A_dflat, rowA * com * batchSize * sizeof(float)));
	cudacall(cudaMalloc(&B_d, batchSize * sizeof(float *)));
	cudacall(cudaMalloc(&B_dflat, colB * com * batchSize * sizeof(float)));

	A[0] = A_dflat;
	B[0] = B_dflat;
	for (int i = 1; i < batchSize; i++)
	{
		A[i] = A[i - 1] + (rowA * com);
		B[i] = B[i - 1] + (colB * com);
	}
	cudacall(cudaMemcpy(A_d, A, batchSize * sizeof(float *), cudaMemcpyHostToDevice));
	cudacall(cudaMemcpy(B_d, B, batchSize * sizeof(float *), cudaMemcpyHostToDevice));

	for (int i = 0; i < batchSize; i++)
	{
		cudacall(cudaMemcpy(A_dflat + (i * rowA * com), src1[i], rowA * com * sizeof(float), cudaMemcpyHostToDevice));
		cudacall(cudaMemcpy(B_dflat + (i * colB * com), src2[i], colB * com * sizeof(float), cudaMemcpyHostToDevice));
	}

	float **C = new float *[batchSize];
	float **C_d, *C_dflat;

	cudacall(cudaMalloc(&C_d, batchSize * sizeof(float *)));
	cudacall(cudaMalloc(&C_dflat, rowA * colB * batchSize * sizeof(float)));
	C[0] = C_dflat;
	for (int i = 1; i < batchSize; i++)
		C[i] = C[i - 1] + (rowA * colB);
	cudacall(cudaMemcpy(C_d, C, batchSize * sizeof(float *), cudaMemcpyHostToDevice));

	//initialize timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Do the actual multiplication
	cublascall(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, colB, rowA, com, &alpha, (const float **)B_d, lda, (const float **)A_d, ldb, &beta, C_d, ldc, batchSize));

	//Stop and record timers
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timer_1, start, stop);

	for (int i = 0; i < batchSize; i++)
		cudacall(cudaMemcpy(dist[i], C_dflat + (i * rowA * colB), rowA * colB * sizeof(float), cudaMemcpyDeviceToHost));

	// Destroy the handle
	cublasDestroy(handle);

	cudaFree(A_d);
	cudaFree(A_dflat);
	cudaFree(B_d);
	cudaFree(B_dflat);
	cudaFree(C_d);
	cudaFree(C_dflat);

	return timer_1;
}

void printArray(float *A, int rowA, int colA, int batchsize, int index = 0, char c = 'A')
{
	fprintf(stdout, "\n%c", c);
	for (int i = 0; i < rowA; i++)
	{
		for (int j = 0; j < colA; j++)
			fprintf(stdout, "%f\t", A[i * colA + j]);
		fprintf(stdout, "\n");
	}
}
void testMatMulBatched()
{
	int mybatch = 3;
	const int rowA = 5, colA = 5;
	const int rowB = colA, colB = 5;
	float timer;

	float **A = new float *[mybatch];
	float **B = new float *[mybatch];
	float **C = new float *[mybatch];

	for (int i = 0; i < mybatch; i++)
	{
		A[i] = new float[rowA * colA];
		B[i] = new float[rowB * colB];
		C[i] = new float[rowA * colB];

		for (int j = 0; j < rowA * colA; j++)
			A[i][j] = (double)rand() / RAND_MAX;
		for (int k = 0; k < rowB * colB; k++)
			B[i][k] = (double)rand() / RAND_MAX;
	}
	timer = MatMulBatched(A, B, C, rowA, colB, colA, mybatch);

	//printing input/output Arrays

	printArrayBatch(A, rowA, colA, mybatch, 'A');
	printArrayBatch(B, rowB, colB, mybatch, 'B');
	printArrayBatch(C, rowA, colB, mybatch, 'C');

	fprintf(stdout, "\n\nTotal time taken for the whole batch: %f us\n\n", timer * 1000);
}

void BackSlashOp()
{
	// Dimensions: A(m x k) ------ A_transpose(k x m) ------- y(m x n)
	int mybatch = 1;
	const int rowA = 3, colA = 3;
	const int rowA_t = colA, colA_t = rowA, rowY = rowA, colY = 1;
	float timer = 0;

	float **A = new float *[mybatch];
	float **AxA_trans = new float *[mybatch];
	float **A_trans = new float *[rowA * colA];
	float **y = new float *[rowA * colA];

	for (int i = 0; i < mybatch; i++)
	{
		A[i] = new float[rowA * colA];
		A_trans[i] = new float[rowA_t * colA_t];
		y[i] = new float[rowY * colY];
		AxA_trans[i] = new float[colA * colA];

		for (int j = 0; j < rowA * colA; j++)
			A[i][j] = (double)rand() / RAND_MAX;
		for (int k = 0; k < rowY * colY; k++)
			y[i][k] = (double)rand() / RAND_MAX;
	}

	printArrayBatch(A, rowA, colA, mybatch, 'A');

	float tTrans = 0;
	for (int i = 0; i < mybatch; i++)
		tTrans += Transpose(A[i], A_trans[i], rowA, colA);

	//printArrayBatch(A_trans, rowA_t, colA_t, mybatch, 'T');

	//C(k,k) =A_trans(k,m) *  A(m,k)
	timer += MatMulBatched(A_trans, A, AxA_trans, rowA_t, colA, rowA, mybatch);
	//MatMulBatched(A_trans,A,C,k,k,m);

	//printArrayBatch(AxA_trans, colA, colA, mybatch, 'C');

	float **invResult = new float *[mybatch];
	float **V = new float *[mybatch];
	float **finalResult = new float *[mybatch];

	for (int i = 0; i < mybatch; i++)
	{
		invResult[i] = new float[colA * colA];
		V[i] = new float[colA * rowA];
		finalResult[i] = new float[colA * colY];
	}

	float timerGPU = -1;
	float timerTotal = -1;

	invert(AxA_trans, invResult, colA, mybatch, timerGPU, timerTotal);

	//printArrayBatch(invResult, colA, colA, mybatch, 'P');

	//V(k,m) =R(k,k) * A_trans(k,m)
	timer += MatMulBatched(invResult, A_trans, V, colA, colA_t, colA, mybatch);

	//printArrayBatch(V, colA, rowA, mybatch, 'V');
	printArrayBatch(y, rowY, colY, mybatch, 'Y');

	//reuslt(m,n) =V(k,m) * y(m,n)
	timer += MatMulBatched(V, y, finalResult, colA, colY, rowA, mybatch);

	printArrayBatch(finalResult, colA, colY, mybatch, 'Z');

	//printing statistics
	fprintf(stdout, "\nCode statistics [computation time only]: ", tTrans);
	fprintf(stdout, "\nTotal time for Transposing: %f ms", tTrans);
	fprintf(stdout, "\nTotal time for all multiplications: %f ms", timer);
	fprintf(stdout, "\nTotal time for Inverse operation: %f ms", timerGPU);

	fprintf(stdout, "\n\n Total timer for BackSlashOp w/o trans: %f ms \n\n\n", tTrans + timer + timerGPU);
}
int main()
{
	srand(time(NULL));
	BackSlashOp();
	return 0;
}