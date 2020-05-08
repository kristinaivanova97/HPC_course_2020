//
//  LeastSquares.c
//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    
    int N = 100; // size of problem;
    double *X;
    double *Y;
    double a = 2;
    double b = 1;
    int i;
    
    X = (double *) malloc(N * sizeof(double));
    Y = (double *) malloc(N * sizeof(double));
    
    #pragma omp parallel // fill the arrays
    {
    int tid = omp_get_thread_num();
    unsigned int seed = tid*12345678;
    #pragma omp for
        for (int i=0;i<N;i++) {
            X[i]= rand_r(&seed) % 100;
            Y[i] = X[i]*a + b + 0.1*((double)rand_r(&seed)/(double)RAND_MAX);
            //printf("x[i] = %f, y[i] = %f, i = %d", X[i], Y[i], i);
        }
    }
    
    //start the algorithm
    double sumOfX = 0;
    double sumOfY = 0;
    double sumOfXSq = 0;
    double sumOfYSq = 0;
    double sumOfXY = 0;
    double rSquared;
    
    
    #pragma omp parallel shared(X, Y)
    {
    double x, y;
    #pragma omp for reduction(+:sumOfXY,sumOfX, sumOfY, sumOfXSq, sumOfYSq )
    for (i = 0; i < N; i++)
    {
        x = X[i];
        y = Y[i];
        sumOfXY += x * y;
        sumOfX += x;
        sumOfY += y;
        sumOfXSq += x * x;
        sumOfYSq += y * y;
    }
    }

    double a_hat;
    double b_hat;
    double ssX = sumOfXSq - ((sumOfX * sumOfX) / N);
    double ssY = sumOfYSq - ((sumOfY * sumOfY) / N);

    double rNumerator = (N * sumOfXY) - (sumOfX * sumOfY);
    double rDenom = (N * sumOfXSq - (sumOfX * sumOfX)) * (N * sumOfYSq - (sumOfY * sumOfY));
    double sCo = sumOfXY - ((sumOfX * sumOfY) / N);

    double meanX = sumOfX / N;
    double meanY = sumOfY / N;
    double dblR = rNumerator / sqrt(rDenom);

    rSquared = dblR * dblR;
 
    b_hat = meanY - ((sCo / ssX) * meanX);
    a_hat = sCo / ssX;
    printf("b_hat = %f, a_hat = %f\n ,where initial values were b = %f, a = %f\n", b_hat, a_hat, b, a);
 
/*
    double a_hat;
    double b_hat;
    double meanX = sumOfX / N;
    double meanY = sumOfY / N;
    b_hat = (N*sumOfXY - sumOfX * sumOfY)*(N*sumOfXSq - sumOfX * sumOfX);
    a_hat = meanY - b_hat*meanX;
    printf("b = %f, a = %f\n", b_hat, a_hat);
 */
    
}
