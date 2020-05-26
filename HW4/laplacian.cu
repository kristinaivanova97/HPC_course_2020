#include <stdio.h>
#include <omp.h>

__global__
void laplace(float * U1, float * U2) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int raw = blockDim.x + 2;
    U2[(i + 1) * raw + j + 1] = U1[i * raw + j + 1] + U1[(i + 1) * raw + j]
	 + U1[(i + 2) * raw + j + 1] + U1[(i + 1) * raw + j + 2]; 
    U2[(i + 1) * raw + j + 1] *= .25;
}

int main() {
    int T = 1000;
    int raw = 1024;
    int grid = raw * raw;

    float * U1, * U2, * devU1, * devU2;
     
    U1 = (float *)malloc(grid * sizeof(float));
    U2 = (float *)malloc(grid * sizeof(float));
    cudaMalloc(&devU1, grid * sizeof(float));
    cudaMalloc(&devU2, grid * sizeof(float));

    for (int i=0; i<raw; ++i)
        U1[i] = 1.;
    //for (int i = 0; i<side; ++i)
//	printf("%f", U1[i]);
    for (int i=1; i<raw; ++i) {
        for (int j=0; j<raw; ++j) 
            U1[i * raw + j] = 0.;
    }
    memcpy(U2, U1, grid * sizeof(float));

    double start = omp_get_wtime();

    cudaMemcpy(devU1, U1, grid * sizeof(float), 
                                 cudaMemcpyHostToDevice);
    cudaMemcpy(devU2, U1, grid * sizeof(float), 
                                 cudaMemcpyHostToDevice);

    for (int t=0; t<T;) { 
        laplace<<<raw-2, raw-2>>>(devU1, devU2);
	laplace<<<raw-2, raw-2>>>(devU2, devU1);
        t += 2;
    }
    cudaMemcpy(U1, devU1, grid * sizeof(float), 
                                 cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    double end = omp_get_wtime();
    printf("%f sec \n", end - start); 
    cudaFree(devU1);
    cudaFree(devU2);
    free(U1);
    free(U2);
}

