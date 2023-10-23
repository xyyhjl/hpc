#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
int main() {
    // 设置矩阵的维度
    int m = 2048;
    int n = 2048;
    int k = 23048; 

    // 分配并初始化CPU上的输入矩阵
    float* A = (float*)malloc(m * k * sizeof(float));
    float* B = (float*)malloc(k * n * sizeof(float));
    for (int i = 0; i < m * k; i++) {
        A[i] = i;
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = i;
    }

    // 在GPU上分配内存
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // 创建cuBLAS句柄
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 创建CUDA事件来测量时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // 执行矩阵乘法
    float alpha = 1.0;
    float beta = 0.0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
    // 停止计时器
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 输出运行时间
    printf("GPU:Matrix multiplication took %f milliseconds.\n", milliseconds);

    // 将结果从设备复制到主机
    float* C = (float*)malloc(m * n * sizeof(float));
    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    //实现朴素矩阵乘法
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);
    for(int i=0;i<m;i++)
    {
    	for(int j=0;j<n;j++)
    	{
    		for(int p=0;p<k;p++)
    		{
    			C[i*n+j]+=A[i*k+p]*B[p*n+j];
    		}

    	}
    }
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start2, stop2);
    printf("Naive:Matrix multiplication took %f milliseconds.\n", milliseconds2);
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    return 0;
}