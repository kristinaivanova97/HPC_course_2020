#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int nThreads = 4;
unsigned int seeds[nThreads];

void seedThreads() {
    int my_thread_id;
    unsigned int seed;
    #pragma omp parallel private (seed, my_thread_id)
    {
        my_thread_id = omp_get_thread_num();
        unsigned int seed = (unsigned) time(NULL);
        seeds[my_thread_id] = (seed & 0xFFFFFFF0) | (my_thread_id + 1);
        
        printf("Thread %d has seed %u\n", my_thread_id, seeds[my_thread_id]);
    }
    
}

int main()
{
    srand((unsigned int)time(NULL));
    const size_t N = 10000000;
    unsigned int N_inside = 0;
    unsigned int counter = 0;
    unsigned int cond, i;
    unsigned int tid;
    unsigned int seed;
    //unsigned int seed = 12345678;

    double x, y, pi;
    omp_set_num_threads(nThreads);
    seedThreads();

    #pragma omp parallel firstprivate(counter) shared(seeds) private(x,y, tid, seed) //reduction(+:N_inside)
       {
           //unsigned int seed = (unsigned) time(NULL);
           //unsigned int seed_safe;
           tid = omp_get_num_threads();
           seed = seeds[tid];
           srand(seed);
           //seed_safe = (seed & 0xFFFFFFF0) | (tid + 1);
           //srand(seed_safe);
           #pragma omp for
           for (i=0; i<N; ++i) {
               x = (double)rand_r(&seed) / RAND_MAX;
               y = (double)rand_r(&seed) / RAND_MAX;
               cond = (x * x + y * y < 1);
               counter += cond;
               #pragma omp atomic update
               N_inside += cond;
           }
           printf("counter = %d\n", counter);
       }

    printf("sum of counters : %d\n", N_inside);
    pi =(double)4 * N_inside / N;
    printf("pi = %.16f\n", pi);

    return 0;
}
