/*
   compile: mpicc -o qsort qsort.c
   run:     mpirun -np num_procs qsort in_file out_file
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// swap entries in array at positions i and j
void swap(int * input, int i, int j)
{
  int tmp = input[i];
  input[i] = input[j];
  input[j] = tmp;
}

// sort part of array input; part starts at s and is of length n
void quicksort(int * input, int s, int n)
{
  int x, place, i;
  // end of algorithm
  if (n <= 1)
    return;
  // pick pivot and swap with first element
  x = input[s + n/2]; //median
  swap(input, s, s + n/2);
  // partition starting at s+1
  place = s;
  for (i = s+1; i < s+n; i++)
    if (input[i] < x) {
      place++;
      swap(input, i, place); // ignore element if it is larger, but place is not moved by 1
        // we devide part of array into 2 groups, in first part < x, in second > x
    }
  // swap pivot into place
  swap(input, s, place);
  // recurse into partition
  quicksort(input, s, place-s); //right part
  quicksort(input, place+1, s+n-place-1); //left part
}


// merge two sorted arrays arrA, arrB of lengths n1, n2
int * merge(int * arrA, int n1, int * arrB, int n2)
{
  int * result = (int *)malloc((n1 + n2) * sizeof(int));
  int i = 0;
  int j = 0;
  int k;
  for (k = 0; k < n1 + n2; k++) {
    if (i >= n1) {
      result[k] = arrB[j];
      j++;
    }
    else if (j >= n2) {
      result[k] = arrA[i];
      i++;
    }
    else if (arrA[i] < arrB[j]) { // indices i < n1 && j < n2
      result[k] = arrA[i];
      i++;
    }
    else { // arrB[j] <= arrA[i]
      result[k] = arrB[j];
      j++;
    }
  }
  return result;
}

int main(int argc, char ** argv)
{
  int n, world_size, id; //array size, num of processors, id of processor
  int chunk_size, s, i; // s is size of local chunck
  int * data = NULL;
  int * chunk;
  int size_recieve;
  int * received_chunks;
  int step;
  MPI_Status status;
  double elapsed_time, startTime;
  FILE * file = NULL;
    
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
    
    if (argc>1)
        {
            if (argc!=3) {
            fprintf(stderr, "Usage: mpirun -np <num_procs> %s <in_file> <out_file>\n", argv[0]);
            exit(1);
            }
    }
    /*if (id == 0) {
      // read size of data
      file = fopen(argv[1], "r");
      fscanf(file, "%d", &n);
      // compute chunk size
      chunk_size = (n%world_size!=0) ? n/world_size+1 : n/world_size;
      // read data from file
      data = (int *)malloc(world_size*chunk_size * sizeof(int));
      for (i = 0; i < n; i++)
        fscanf(file, "%d", &(data[i]));
      fclose(file);
      // pad data with 0 -- doesn't matter
      for (i = n; i < world_size*chunk_size; i++)
        data[i] = 0;
    }
     */
    
    // *** Process 0 initialize data ***
    if (id == 0){
        if (argc > 1)
        {
            file = fopen(argv[1],"r");
            printf("OPEN FILE\n");
            //first number in the input file should specify the total number of elements in the input array
            //the rest of the numbers in the input file are the elements of the array, which is filled in row major order
            fscanf(file,"%d",&n);
            //master node reads the input file
            chunk_size = (n % world_size != 0) ? n / world_size + 1 : n / world_size;
            // read data from file
            data = (int *)malloc(world_size*chunk_size * sizeof(int));
            for (i = 0; i < n; i++) fscanf(file, "%d", &(data[i]));
            fclose(file);
            // pad data with 0 to have equal data for each processor
            for (i = n; i < world_size*chunk_size; i++) data[i] = 0;
        }
            else
            {
                n = 1048577;
                if (id == 0)
                {
                    srand(time(NULL));
                    chunk_size = (n % world_size != 0) ? n / world_size + 1 : n / world_size;
                    data = (int*)malloc(world_size*chunk_size*sizeof(int));
                    for (int i = 0; i < n; i++) {
                        data[i] = rand() % 100;
                        
                    }
                    // pad data with 0 to have equal data for each processor
                    for (i = n; i < world_size*chunk_size; i++)
                      data[i] = 0;
                    
                }
            }
        
    }
    

  MPI_Barrier(MPI_COMM_WORLD);
  elapsed_time = - MPI_Wtime();

  // broadcast size
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // scatter data
  chunk = (int *)malloc(chunk_size * sizeof(int));
  MPI_Scatter(data, chunk_size, MPI_INT, chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
  free(data);
  data = NULL;

  // compute size of own chunk and sort it
  s = (n >= chunk_size * (id+1)) ? chunk_size : n - chunk_size * id;
  quicksort(chunk, 0, s);

  // up to log_2 world_size merge steps
  for (step = 1; step < world_size; step = 2*step) {
    if (id % (2*step) != 0) {
      // if id is not multiple of 2*step: send chunk to id - step and exit loop
      MPI_Send(chunk, s, MPI_INT, id-step, 0, MPI_COMM_WORLD);
      break;
    }
    // id is multiple of 2*step: merge in chunk from id+step (if it exists)
    if (id+step < world_size) {
      // compute size of chunk to be received
      size_recieve = (n >= chunk_size * (id+2*step)) ? chunk_size * step : n - chunk_size * (id+step);
      // received_chunks
      received_chunks = (int *)malloc(size_recieve * sizeof(int));
      MPI_Recv(received_chunks, size_recieve, MPI_INT, id+step, 0, MPI_COMM_WORLD, &status);
      // merge and free memory
      data = merge(chunk, s, received_chunks, size_recieve);
      free(chunk);
      free(received_chunks);
      chunk = data;
      s = s + size_recieve;
    }
  }

  // stop the timer
  MPI_Barrier(MPI_COMM_WORLD);
  elapsed_time += MPI_Wtime();

    if (id == 0) {
        /*for (i = 0; i < s; i++)
            printf("%d ", chunk[i]);
        printf("\n");
         */
        printf("Quicksort %d ints on %d procs: %f secs\n", n, world_size, elapsed_time);
    }
    printf("\n");
  // write sorted data to out file and print out timer
    if (argc > 2){
        if (id == 0) {
            file = fopen(argv[2], "w");
            fprintf(file, "%d\n", s);   // assert (s == n)
            for (i = 0; i < s; i++)
                fprintf(file, "%d\n", chunk[i]);
            fclose(file);
            
        }
    }

  MPI_Finalize();
  return 0;
}
