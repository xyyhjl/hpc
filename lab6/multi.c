#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
typedef struct {
    int n;
    double *A;
    double *B;
    double *C;
    int block_size;
    int start_row;
    int end_row;
} ThreadData;

void *dgemm_block_thread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    int n = data->n;
    double *A = data->A;
    double *B = data->B;
    double *C = data->C;
    int block_size = data->block_size;
    int start_row = data->start_row;
    int end_row = data->end_row;

    for (int i = start_row; i < end_row; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int k = 0; k < n; k += block_size) {
                // 对每个block进行矩阵乘法
                for (int ii = i; ii < i + block_size; ii++) {
                    for (int jj = j; jj < j + block_size; jj++) {
                        for (int kk = k; kk < fmin(k + block_size,n); kk++) {
                            C[ii * n + jj] += A[ii * n + kk] * B[kk * n + jj];
                        }
                    }
                }
            }
        }
    }

    pthread_exit(NULL);
}

void dgemm_block_multithread(int n, double *A, double *B, double *C, int block_size, int num_threads) {
    pthread_t threads[num_threads];
    ThreadData data[num_threads];

    int rows_per_thread = n / num_threads;

    for (int t = 0; t < num_threads; t++) {
        data[t].n = n;
        data[t].A = A;
        data[t].B = B;
        data[t].C = C;
        data[t].block_size = block_size;
        data[t].start_row = t * rows_per_thread;
        data[t].end_row = (t + 1) * rows_per_thread;

        if (t == num_threads - 1) {
            // 最后一个线程处理剩余的行
            data[t].end_row = n;
        }

        pthread_create(&threads[t], NULL, dgemm_block_thread, (void *)&data[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

int main() {
    int n = 2000; // 矩阵维度
    int block_size = 4; // 分块大小
    int num_threads = 10; // 线程数量

    // 初始化矩阵A、B、C
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *C = (double *)calloc(n * n, sizeof(double));

    for (int i = 0; i < n * n; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    // 调用多线程的DGEMM分块函数
    dgemm_block_multithread(n, A, B, C, block_size, num_threads);

    //
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
