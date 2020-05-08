// useful article for Gauss-Seidel for Dirichle equation
// here performed GS for linear equation A*x = b
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void fill( int n, double **A, double *b );

void run_gauss_seidel_method( int p, int n, double **A, double *b,
   double epsilon, int maxit,
   int *numit, double *x );

int main ( int argc, char *argv[] )
{
   int n,p,i;
   if(argc > 1)
   {
      n = atoi(argv[1]);
      p = (argc > 2) ? atoi(argv[2]) : 1;
   }
   else
   {
      printf("give the dimension : ");
      scanf("%d",&n);
      printf("Give the number of threads : ");
      scanf("%d",&p);
   }
   omp_set_num_threads(p);
   {
      double *b;
      b = (double*) calloc(n,sizeof(double));
      double **A;
      A = (double**) calloc(n,sizeof(double*));
      for(i=0; i<n; i++)
         A[i] = (double*) calloc(n,sizeof(double));
       
      fill(n,A,b);
       
      double *x;
      x = (double*) calloc(n,sizeof(double));

      for(i=0; i<n; i++) x[i] = 0.0; //initialize with zeros
      double eps = 1.0e-4;
      int maxit = 300;
      int cnt = 0;
       
      double mo =- omp_get_wtime();
       
      run_gauss_seidel_method(p,n,A,b,eps,maxit,&cnt,x);
       
      mo += omp_get_wtime();
       
      printf("computed %d iterations\n",cnt);
      double sum = 0.0;
      for(i=0; i<n; i++) //compute the error
      {
         double d = x[i] - 1.0;
         sum += (d >= 0.0) ? d : -d;
      }
      printf("error : %.3e\n",sum);
       printf("elapsed time: %f s\n", mo);
   }
   return 0;
}

void fill( int n, double **A, double *b )
{
   int i,j;
   for(i=0; i<n; i++)
   {
      b[i] = 2.0*n;
      for(j=0; j<n; j++) A[i][j] = 1.0;
      A[i][i] = n + 1.0;
   }
}

void run_gauss_seidel_method( int p, int n, double **A, double *b,
   double epsilon, int maxit,
   int *numit, double *x )
{
   double *dx;
   dx = (double*) calloc(n,sizeof(double));
   int i,j,k,id,jstart,jstop;

   int dnp = n/p;
    int res = n%p;
    printf("res of division amoung threads = %d\n", res);
   double dxi;

   for(k=0; k<maxit; k++)
   {
      double sum = 0.0;
      for(i=0; i<n; i++)
      {
         dx[i] = b[i];
         #pragma omp parallel shared(A,x, n) private(id,j,jstart,jstop,dxi)
         {
            id = omp_get_thread_num();
            jstart = id*(dnp+1);
            jstop = jstart + (dnp+1);
             if (jstop > n) {jstop = jstop - (jstop-n);} //not to lose any bit of information
            dxi = 0.0;
            for(j=jstart; j<jstop; j++)
               dxi += A[i][j]*x[j]; 
            #pragma omp critical
               dx[i] -= dxi;
         }
         dx[i] /= A[i][i];
         x[i] += dx[i];
         sum += ( (dx[i] >= 0.0) ? dx[i] : -dx[i]);
      }
      printf("%4d : %.3e\n",k,sum);
      if(sum <= epsilon) break;
   }
   *numit = k+1;
   free(dx);
}

// as a result i've got a constant value of iterations needed for convergence when number of threads is changing. So, no race conditions are detected.
