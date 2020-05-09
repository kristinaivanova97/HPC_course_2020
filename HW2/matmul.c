#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void zero_init_matrix(double ** matrix, size_t N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = 0.0;
        }
    }
}

void rand_init_matrix(double ** matrix, size_t N)
{
    srand(time(NULL));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = rand() / RAND_MAX;
        }
    }
}

double ** malloc_matrix(size_t N)
{
    double ** matrix = (double **)malloc(N * sizeof(double *));
    
    for (int i = 0; i < N; ++i)
    {   
        matrix[i] = (double *)malloc(N * sizeof(double));
    }
    
    return matrix;
}

void free_matrix(double ** matrix, size_t N)
{
    for (int i = 0; i < N; ++i)
    {   
        free(matrix[i]);
    }
    
    free(matrix);
}
void main_ijn();
void main_jin();

int main()
{
    const size_t N = 1000; // size of an array

    //clock_t start, end;
 
    double ** A, ** B, ** C; // matrices
    
    int i, j, n;

    printf("Starting:\n");

    A = malloc_matrix(N);
    B = malloc_matrix(N);
    C = malloc_matrix(N);    

    rand_init_matrix(A, N);
    rand_init_matrix(B, N);
    zero_init_matrix(C, N);

    //start = clock();

//
//  matrix multiplication algorithm
//
    double mo =- omp_get_wtime();
    #pragma omp parallel shared(A,B,C) private(i,j,n)
       {
    #pragma omp for schedule(static)
       for (n=0; n<N; ++n){
          for (i=0; i<N; ++i){
             for (j=0; j<N; ++j){
    #pragma omp atomic update
                C[i][j]+=(A[i][n])*(B[n][j]);
                
             }
           }
       }
       }
       

    mo += omp_get_wtime();

    //printf("Time elapsed (nij): %f seconds.\n", (double)(end - start) / CLOCKS_PER_SEC);
    printf("Time elapsed (nij): %f seconds.\n", mo);

    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);
    main_ijn();
    main_jin();
    // nij loops showed the best performance (6 seconds), then ijn and the worst is jin (16 seconds)
    return 0;
}

void main_ijn()
{
        const size_t N = 1000; // size of an array
        
     
        double ** A, ** B, ** C; // matrices
        
        int i, j, n;

        printf("Starting:\n");

        A = malloc_matrix(N);
        B = malloc_matrix(N);
        C = malloc_matrix(N);

        rand_init_matrix(A, N);
        rand_init_matrix(B, N);
        zero_init_matrix(C, N);

        double mo =- omp_get_wtime();
        #pragma omp parallel shared(A,B,C) private(i,j,n)
           {
        #pragma omp for schedule(static)
           for (i=0; i<N; ++i){
              for (j=0; j<N; ++j){
                 for (n=0; n<N; ++n){
                    C[i][j]+=(A[i][n])*(B[n][j]);
                    
                 }
               }
           }
           }
        mo += omp_get_wtime();
    
        printf("Time elapsed (ijn): %f seconds.\n", mo);

        free_matrix(A, N);
        free_matrix(B, N);
        free_matrix(C, N);
}

void main_jin()
{
        const size_t N = 1000; // size of an array
     
        double ** A, ** B, ** C; // matrices
        
        int i, j, n;

        printf("Starting:\n");

        A = malloc_matrix(N);
        B = malloc_matrix(N);
        C = malloc_matrix(N);

        rand_init_matrix(A, N);
        rand_init_matrix(B, N);
        zero_init_matrix(C, N);
    
        double mo =- omp_get_wtime();

        #pragma omp parallel shared(A,B,C) private(i,j,n)
           {
        #pragma omp for schedule(static)
           for (j=0; j<N; ++j){
              for (i=0; i<N; ++i){
                 for (n=0; n<N; ++n){
                    C[i][j]+=(A[i][n])*(B[n][j]);
                    
                 }
               }
           }
           }
        mo += omp_get_wtime();

        printf("Time elapsed (jin): %f seconds.\n", mo);

        free_matrix(A, N);
        free_matrix(B, N);
        free_matrix(C, N);
}
