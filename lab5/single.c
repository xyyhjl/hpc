#include <stdio.h>
#include <stdlib.h>

void dgemm_block(int n, double *A, double *B, double *C, int block_size) {
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int k = 0; k < n; k += block_size) {
                // 对每个block进行矩阵乘法
                for (int ii = i; ii < i + block_size; ii++) {
                    for (int jj = j; jj < j + block_size; jj++) {
                        for (int kk = k; kk < k + block_size; kk++) {
                            C[ii * n + jj] += A[ii * n + kk] * B[kk *n + jj];
                        }
                    }
                }
            }
        }
    }
}

int main() {
    int n = 2000; // 矩阵维度
    int block_size = 4; // 分块大小

    // 初始化矩阵A、B、C
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *C = (double *)calloc(n * n, sizeof(double));

    for (int i = 0; i < n * n; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    // 调用单线程的DGEMM分块函数
    dgemm_block(n, A, B, C, block_size);

    // 打印结果矩阵C
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", C[i * n + j]);
        }
        printf("\n");
    }

    // 释放内存
    free(A);
    free(B);
    free(C);

    return 0;
}
