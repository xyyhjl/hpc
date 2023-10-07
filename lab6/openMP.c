/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <omp.h>
typedef struct {
    int n;
    double *A;
    double *B;
    double *C;
    int block_size;
    int start_row;
    int end_row;
} ThreadData;

void AddDot( int, double *, int, double *, double * );

void *dgemm_block_thread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    int n = data->n;
    double *A = data->A;
    double *B = data->B;
    double *C = data->C;
    int block_size = data->block_size;
    int start_row = data->start_row;
    int end_row = data->end_row;
    #pragma omp parallel for num_threads(10)
    for (int i = start_row; i < end_row; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int k = 0; k < n; k += block_size) {
                // 对每个block进行矩阵乘法
                for (int ii = i; ii < fmin(i + block_size,n); ii++) {
                    for (int jj = j; jj < fmin(j + block_size,n); jj++) {
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
/* Create macro to let X( i ) equal the ith element of x */

#define X(i) x[ (i)*incx ]

void AddDot( int k, double *x, int incx,  double *y, double *gamma )
{
  /* compute gamma := x' * y + gamma with vectors x and y of length n.

     Here x starts at location x with increment (stride) incx and y starts at location y and has (implicit) stride of 1.
  */

  int p;

  for ( p=0; p<k; p++ ){
    *gamma += X( p ) * y[ p ];
  }
}
