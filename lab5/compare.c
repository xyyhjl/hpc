#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>

// Naive DGEMM函数
void dgemm_naive(int n, double *A, double *B, double *C) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// 分块多线程DGEMM函数
typedef struct {
    int n;
    int block_size;
    int num_threads;
    double *A;
    double *B;
    double *C;
} dgemm_args;

void *dgemm_block_multithread_worker(void *arg) {
    dgemm_args *args = (dgemm_args *)arg;
    int n = args->n;
    int block_size = args->block_size;
    int num_threads = args->num_threads;
    double *A = args->A;
    double *B = args->B;
    double *C = args->C;

    int num_blocks = n / block_size;
    int block_start, block_end;
    int thread_id = 0;

    // 计算每个线程需要计算的块范围
    while (thread_id < num_threads) {
        block_start = thread_id * num_blocks / num_threads;
        block_end = (thread_id + 1) * num_blocks / num_threads;

        // 计算每个块的乘法
        for (int i = block_start; i < block_end; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    for (int x = i * block_size; x < (i + 1) * block_size; x++) {
                        C[x * n + j] += A[x * n + k] * B[k * n + j];
                    }
                }
            }
        }

        thread_id++;
    }

    pthread_exit(NULL);
}

void dgemm_block_multithread(int n, double *A, double *B, double *C, int block_size, int num_threads) {
    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    dgemm_args *args = (dgemm_args *)malloc(num_threads * sizeof(dgemm_args));

    // 创建线程并分配参数
    for (int i = 0; i < num_threads; i++) {
        args[i].n = n;
        args[i].block_size = block_size;
        args[i].num_threads = num_threads;
        args[i].A = A;
        args[i].B = B;
        args[i].C = C;

        pthread_create(&threads[i], NULL, dgemm_block_multithread_worker, &args[i]);
    }

    // 等待所有线程完成
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(args);
}

// 获取当前时间的微秒数
long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000LL + tv.tv_usec;
}

int main() {
    int n = 1000;//矩阵维度
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

    // 测量naive dgemm的执行时间
    long long start_time = get_time();
    dgemm_naive(n, A, B, C);
    long long end_time = get_time();
    printf("Naive DGEMM执行时间：%lld微秒\n", end_time - start_time);

    // 重新初始化矩阵C
    for (int i = 0; i < n * n; i++) {
        C[i] = 0.0;
    }

    // 测量分块多线程dgemm的执行时间
    start_time = get_time();
    dgemm_block_multithread(n, A, B, C, block_size, num_threads);
    end_time = get_time();
    printf("分块多线程DGEMM执行时间：%lld微秒\n", end_time - start_time);

    // 释放内存
    free(A);
    free(B);
    free(C);

    return 0;
}
