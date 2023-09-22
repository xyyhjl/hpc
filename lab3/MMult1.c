/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

//void AddDot( int, double *, int, double *, double * );
void dgemm_(char*, char*, int*, int*,int*, double*, double*, int*, double*, int*, double*, double*, int*);
void MY_MMult( int m, int n, int k, double *a, int lda,
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  char ta='N';
  char tb='N';
  double beta=0.001;
  double alpha=1.2;
  dgemm_(&ta, &tb, &m, &n, &k, &alpha, a, &m, b, &k, &beta, c, &m);
}

/* Create macro to let X( i ) equal the ith element of x */

#define X(i) x[ (i)*incx ]

/*void AddDot( int k, double *x, int incx,  double *y, double *gamma )
{
  /* compute gamma := x' * y + gamma with vectors x and y of length n.

     Here x starts at location x with increment (stride) incx and y starts at location y and has (implicit) stride of 1.
  */

 // int p;

 // for ( p=0; p<k; p++ ){
   // *gamma += X( p ) * y[ p ];
 //}
//}
