//command line arguments: 1 - leftmost point
//*********************** 2 - rightmost point
//*********************** 3 - total number of points
//*********************** 4 - total number of threads

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>

//function to get the timings
unsigned long get_time()
{
        struct timeval tv;
        gettimeofday(&tv, NULL);
        unsigned long ret = tv.tv_usec;
        ret /= 1000;
        ret += (tv.tv_sec * 1000);
        return ret;
}


//variables to store results of numerical integration
double mutex_res = 0;          // result of calculation using mutex method
double semaphore_res = 0;      // result of calculation using semaphore method
double busy_wait_res = 0;      // result of calculation using busy wait method

// global "flags"
pthread_mutex_t mutex;
sem_t semaphore;
int busy_wait_flag = 0;

// you may need this global variables, but you can make them inside main()
double a;                 // left point
double b;                 // right point
int n;                     // number of discretization points
double h;                 // distance between neighboring discretization points
int TOTAL_THREADS;

//mathematical function that we calculate the arc length (declaration, define it yourselves)
double function(double x);
double derivative(double x);
//function to calculate numerical derivative
double numerical_derivative(double x);

//arc_length on a single thread
double serial_arc_length(int start, int n_local);

//multithreaded arc_length rule using busy waiting
void* busy_wait_arc_length(void*);
void busy_wait_main();

//multithreaded arc_length using mutex
void* mutex_arc_length(void*);
void mutex_main();

//multrthreaded arc_length using semaphore
void* semaphore_arc_length(void*);
void semaphore_main();

int main( int argc, char *argv[] )
{
    a = atoi(argv[1]);
    b = atoi(argv[2]);
    n = atof(argv[3]);
    h = (b-a)/n;
    TOTAL_THREADS = atoi(argv[4]);
    printf("TOTAL NUMBER OF THREADS: %d\n", TOTAL_THREADS);    
    long start = get_time();
    double duration;
    double result = serial_arc_length(0, n);
    duration = (get_time() - start);
    printf("solution on a single thread: %f, time: %f milliseconds\n", result, duration);
    busy_wait_main();    
    mutex_main();
    semaphore_main();
    return 0;
}

double function(double x)
{
    return pow(x, 2);
}
double derivative(double x)
{
    return (function(x + h) - function(x - h)) / (2*h);

}

double serial_arc_length(int start, int n_local)
{
    double ai, bi;
    int i;
    double l = 0;
    
    for (i = start; i < start + n_local; i++) {
        ai = a + i*h;
        bi = ai + h; //neighbouring points
        l += (sqrtf(1 + pow(derivative(ai),2)) \
              + sqrtf(1 + pow(derivative(bi),2)))/2 * h;
    }
    
    return l;
}

void* busy_wait_arc_length(void* rank)
{
    int r = * ((int *)rank);
    int l = (n + TOTAL_THREADS - 1) / TOTAL_THREADS;
    int l_cond = (r + 1) * l > n ? n - r * l : l;
    
    double res = serial_arc_length(r * l , l_cond);
    
    while(busy_wait_flag != r);
    busy_wait_res += res;
    busy_wait_flag++;
    return NULL;
    
}

void* mutex_arc_length(void* rank)
{
    int r = * ((int *)rank);
    int l = (n + TOTAL_THREADS - 1) / TOTAL_THREADS;
    int l_cond = (r + 1) * l > n ? n - r * l : l;
    
    double res = serial_arc_length(r * l , l_cond);
    
    pthread_mutex_lock(&mutex);
    mutex_res += res;
    pthread_mutex_unlock(&mutex);
    return NULL;
}

void* semaphore_arc_length(void *rank)
{
    
    int r = * ((int *)rank);
    int l = (n + TOTAL_THREADS - 1) / TOTAL_THREADS;
    int l_cond = (r + 1) * l > n ? n - r * l : l;
    
    double res = serial_arc_length(r * l , l_cond);
    sem_wait(&semaphore);
    semaphore_res += res;
    sem_post(&semaphore);
    return NULL;
    
}

void busy_wait_main()
{
    pthread_t* thread_ptr;
    thread_ptr = malloc( TOTAL_THREADS * sizeof(pthread_t));

    long start = get_time();
    double duration;

    //
    int i;
    int* ranks;
    ranks = malloc( TOTAL_THREADS * sizeof(int));
    for (i = 0; i < TOTAL_THREADS; i++)
    {
        ranks[i] = i;
        pthread_create(&(thread_ptr[i]), NULL, &busy_wait_arc_length, (void *)(ranks + i));
    }
    for (i = 0; i < TOTAL_THREADS; i++)
    {
        pthread_join(thread_ptr[i], NULL);
    }
    //

    duration = (get_time() - start);
    printf("solution using busy waiting: %f, time: %f milliseconds\n", busy_wait_res, duration);
    free(ranks);
    free(thread_ptr);    
}

void mutex_main()
{
    // write your implementation here
    pthread_t* thread_ptr;
    thread_ptr = malloc( TOTAL_THREADS * sizeof(pthread_t));

    long start = get_time();
    double duration;
    int i;
    int* ranks;
    ranks = malloc( TOTAL_THREADS * sizeof(int));
    pthread_mutex_init(&mutex, NULL);
    for (i = 0; i < TOTAL_THREADS; i++)
    {
        ranks[i] = i;
        pthread_create(&(thread_ptr[i]), NULL, &mutex_arc_length, (void *)(ranks + i));
    }
    for (i = 0; i < TOTAL_THREADS; i++)
    {
        pthread_join(thread_ptr[i], NULL);
    }
    //
    pthread_mutex_destroy(&mutex);
    duration = (get_time() - start);
    printf("solution using mutex: %f, time: %f milliseconds\n", mutex_res, duration);

    free(thread_ptr);
    free(ranks);
}

void semaphore_main()
{
    // write your implementation here
    pthread_t* thread_ptr;
    thread_ptr = malloc( TOTAL_THREADS * sizeof(pthread_t));

    long start = get_time();
    double duration;
    int i;
    int* ranks;
    ranks = malloc( TOTAL_THREADS * sizeof(int));
    sem_init(&semaphore, NULL, 1);
    for (i = 0; i < TOTAL_THREADS; i++)
    {
        ranks[i] = i;
        pthread_create(&(thread_ptr[i]), NULL, &semaphore_arc_length, (void *)(ranks + i));
    }
    for (i = 0; i < TOTAL_THREADS; i++)
    {
        pthread_join(thread_ptr[i], NULL);
    }
    sem_destroy(&semaphore);

    duration = (get_time() - start);
    printf("solution using semaphore: %f, time: %f milliseconds\n", semaphore_res, duration);

    free(thread_ptr);
    free(ranks);
}
