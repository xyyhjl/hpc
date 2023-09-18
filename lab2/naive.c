#include <stdio.h>
#include "time.h"
#include <stdlib.h>
int M=0;
int N=0;
int K=0;


void dgemm(int m, int n, int k, int beta,
          double A[][k], double B[][n], double C[][n]){
    for(int i=0; i< m;i ++){    //C[i] 
        for(int j=0; j< n; j++){  //C[i][j]
            C[i][j] = beta*C[i][j];
            for(int p=0; p< k; p++){  
                C[i][j] += A[i][p]*B[p][j]; 
             }
        }
    }
}

void printf_matrix(int row, int col, double matrix[row][col] ){
  for(int i=0; i<row; i++){
    for(int j=0; j<col;j++){
        printf("%lf ", matrix[i][j]);
    }
    printf("\n");
  }
  printf("\n\n");
}


int main()
{
    scanf("%d%d%d",&M,&N,&K);
    int sizeofa = M * K;
    int sizeofb = K * N;
    int sizeofc = M * N;
    double A[M][K];
    double B [K][N];
    double C [M][N];

    srand((unsigned)time(NULL));

    for(int i=0;i<M;i++)
    {
        for(int j=0;j<K;j++)
        {
            A[i][j] = ((i-1)*K+j)%3+1;//(rand()%100)/10.0;
        }
    }

    for (int i=0; i<K; i++)
    {
        for(int j=0;j<N;j++)
        {
            B[i][j] = ((i-1)*N+j)%3+1;
        }
    }

    for (int i=0; i<M; i++)
        for(int j=0;j<N;j++)
            C[i][j] = ((i-1)*N+j)%3+1;


    //C=A*B + beta*C
    struct timeval start,finish;
    gettimeofday(&start, NULL);
    dgemm(M,N,K,2, A,B,C);
    gettimeofday(&finish, NULL);
    double gflops = 2.0 * M *N*K;
    double duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
    printf_matrix(M,K,A);
    printf_matrix(K,N,B);
    printf_matrix(M,N,C);
    FILE *fp;
    fp = fopen("timeDgemm.txt", "a");
    fprintf(fp, "%dx%dx%d\t%lf s\t%lf GFLOPS\n", M, N, K, duration, gflops);
    fclose(fp);
    return 0;

}
