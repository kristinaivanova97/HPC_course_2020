// /usr/local/Cellar/open-mpi/4.0.3/bin/mpicc -o radix radixOBlocks.c -lm
// /usr/local/Cellar/open-mpi/4.0.3/bin/mpirun --use-hwthread-cpus -np 4 radix

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

// a random integer in range [min, max]
int randomInRange(int min, int max) {
  return rand() % (max - min + 1) + min;
}

// a maximum integer in an array (input) of length n.
int max(int * input, int n) {
  int tmp = 0;
  for (int i = 0; i < n; i++)
  {
    if (input[i] > tmp) tmp = input[i];
  }
  return tmp;
}
/* Sort the input by the digit in the exp'th place and write in
 * count[] to maintain the histogram of exp'th place digit values for every element (number) in array.
 * inspired by geeksforgeeks.org/radix-sort/
 */
void countingSort(int * input, int * count, int n, int exp) {
    
  int * buckets = malloc(n * sizeof(int));
  // PrefixSum stores the count of values for each digit value, then it store the sum of previous counts
  int PrefixSum[10];
  int i;
  int idx;

  for (i = 0; i < n; i++) {
    count[(input[i] / exp) % 10]++; // to get value on exp'th place for every element and count its amount
  }

  // copy count array to workingCount array
  for (i = 0; i < 10; i++) {
    PrefixSum[i] = count[i];
  }

  /* Here we get prefix sum:
   Each value in PrefixSum is the sum of the previous values
   */
  for (i = 1; i < 10; i++) {
    PrefixSum[i] += PrefixSum[i - 1];
  }

  /* Here is the main step, for each value on exp'th place of every number in input array we have an exact position in buckets
   (first elements in buckets are "0" and their amount is PrefixSum[0] and so on)
   */
  for (i = n - 1; i >= 0; i--) {
    idx = (input[i] / exp) % 10;
    buckets[PrefixSum[idx] - 1] = input[i];
    PrefixSum[idx]--;
  }

  for (i = 0; i < n; i++) {
    input[i] = buckets[i];
  }
  free(buckets);
}


// To check correctness of parallel sorting algorithm.
void compareArrays(int * arrA, int * arrB, int n) {
  for (int i = 0; i < n; i++) {
    if (arrA[i] != arrB[i]) {
        printf("Arrays are not equal!\n");
    }
  }
    //printf("Algorithm is right!\n");
}

/* Function to sort input of length n using radix sort with counting sort subroutine,
    here n - is the length of the local array for each processor.
 */
void RadixSort(int * input, int n) {
  int m = max(input, n); //to check all the places in all numbers
  for (int exp = 1; m / exp > 0; exp *= 10) {
    int count[10] = {0};
    countingSort(input, count, n, exp);
  }
    //for now, we placed numbers in ascending order
}

// Check the correctness of parallel radix sort. Input is an original arr.
void check(int * input, int * sortedInput, int n) {
  int * radix_arr = (int * ) malloc(n * sizeof(int));
  for (int i = 0; i < n; i++) {
    radix_arr[i] = input[i];
  }
  RadixSort(radix_arr, n);
  compareArrays(radix_arr, sortedInput, n);
  free(radix_arr);
}

int main(int argc,const char * argv[])
{
  int world_rank;
  int world_size;
    
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, & world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, & world_size);

  // seeds for random number generator to be used
  int seeds[] = {
    1,
    2,
    22443,
    16882,
    7931,
    10723,
    24902,
    124,
    25282,
    2132
  };
//to average time consumption, the calculations are repeated "runs" number times.
  int runs = 10;

  double elapsed_time;
  double total = 0;
    
    
  // run on array sizes 1, 2, 4 ... 2^20
  for (int size = 1; size < 1048577; size *= 2) {
    MPI_Barrier(MPI_COMM_WORLD);
    for (int run = 0; run < runs; run++) {
      MPI_Barrier(MPI_COMM_WORLD);

      int * input;
      int * adjustedInput; // adjusted array to account for artificially added zeros
      int n; // size of the array
      int nadj; // size of the array + the size of the adjusted array
      int m; // max value in the array
      int local_size; // size of the local array on each processor
      int tailOffset; // number of artificially added zeros
        
      double start = MPI_Wtime();
        
      if (world_rank == 0) {
        srand(seeds[run]);

        n = size; //len of input array for that loop step
        input = (int * ) malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
          //input[j] = randomInRange(0, n);
          // to generate reverse ordered arrays
          // input[j] = n - j - 1;
          // to build random array with exactly one digit
          // input[j] = randomInRange(0, 9);
            // to build random array with exactly eight digits use the line below
             input[j] = randomInRange(10000000, 99999999);
        }

        // compute local_size of local array for each processor
        local_size = ceil(n / ((double) world_size));
        //ceil - function that return the number equal or more than double type argument.

        // number of zeros to add to array to make sure each
        // processor has exactly n/p elements
        tailOffset = world_size * local_size - n;

        // add zeros to an array with new size to account for tailOffset zeros
        nadj = local_size * world_size;
        adjustedInput = (int * ) malloc(nadj * sizeof(int));
        for (int i = 0; i < nadj; i++) {
          if (i < tailOffset) {
            adjustedInput[i] = 0;
          } else {
            adjustedInput[i] = input[i - tailOffset];
          }
        }
        m = max(input, n);
      }
        //computed on 0 process and brodcast to all
      MPI_Bcast( &m, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast( &n, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast( &local_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast( &nadj, 1, MPI_INT, 0, MPI_COMM_WORLD);

      MPI_Barrier(MPI_COMM_WORLD);

      // local array
      int * local_input = (int*)malloc(local_size * sizeof(int));

      // scatter input from 0 to all processors such that each processor holds n/p values in its local_input
      MPI_Scatter(adjustedInput, local_size, MPI_INTEGER, local_input, local_size, MPI_INTEGER, 0, MPI_COMM_WORLD);

      // 2D arrays to hold pointers to blocks to be sent to each processor and to be received from each processor during transpose step
      int ** sendBlocks = malloc(world_size * sizeof(int * ));
      int ** receiveBlocks = malloc(world_size * sizeof(int * ));

      for (int i = 0; i < world_size; i++) {
        sendBlocks[i] = (int * ) malloc(local_size * 2 * sizeof(int));
          // multiple by 2 as we send not only values, but their local index also
        receiveBlocks[i] = (int * ) malloc(local_size * 2 * sizeof(int));
      }

      //*** MAIN BODY ***
        
      for (int exp = 1; m / exp > 0; exp *= 10)
      {
        // set all values in the blocks to -1
        for (int i = 0; i < world_size; i++) {
          memset(sendBlocks[i], -1, local_size * 2 * sizeof(int));
        }

        int count[10] = {0}; // histogram for local_input array
        // GlobalCounts hold the histograms for all processors
        int GlobalCounts[10 * world_size];

        // local array after redistribution step
        int * redistributed = malloc(local_size * sizeof(int));
          
        int GlobalCountsSum[10] = {0}; // histogram for global array
        int GlobalCountsPrefixSum[10] = {0}; // cumulative sum array to be built from GlobalCountsSum
        int GlobalCountsSumLeft[10] = {0}; // histogram for elements "left" of current processor, as we send and recieve keys according to processors id and maintain their position 0 1 2 ... p

        // counting sort, save local histogram to count and share histogram with all other processors
        countingSort(local_input, count, local_size, exp);
        MPI_Allgather(count, 10, MPI_INTEGER, GlobalCounts, 10, MPI_INTEGER, MPI_COMM_WORLD);

        // sum up histograms into global histogram GlobalCountsSum
        for (int i = 0; i < 10 * world_size; i++) {
          int column = i % 10;
          int p = i / 10;
          int val = GlobalCounts[i];

          // add histogram values to GlobalCountsSumLeft for all processors "left" of current processor
          if (p < world_rank) {
            GlobalCountsSumLeft[column] += val; // so for each processor we have number of counts, which go before it - to compute place for its values.
          }
          GlobalCountsSum[column] += val; //how many elements have "column" value
          GlobalCountsPrefixSum[column] += val; //how many elements there are for each key (digit on exp'th position)
        }
        // Global prefix sum
        for (int i = 1; i < 10; i++) {
          GlobalCountsPrefixSum[i] += GlobalCountsPrefixSum[i - 1];
        }

        MPI_Request request;
        MPI_Status status;

        // count of elements to be sent from current processor for each key(digit)
        int Sent_element_idx[10] = {0};

        int val, column, destIndex, destProcess, localDestIndex;

        // number of elements in each block
        int blockSent[10] = {0};

        // build blocks from values
        for (int i = 0; i < local_size; i++) {
          val = local_input[i];
          column = (local_input[i] / exp) % 10;

          destIndex = GlobalCountsPrefixSum[column] - GlobalCountsSum[column] + GlobalCountsSumLeft[column] + Sent_element_idx[column]; //global
          Sent_element_idx[column]++;
          destProcess = destIndex / local_size; //transpose
          localDestIndex = destIndex % local_size;

          // set value, local index in destination process
          sendBlocks[destProcess][blockSent[destProcess] * 2] = val;
          sendBlocks[destProcess][blockSent[destProcess] * 2 + 1] = localDestIndex;
          blockSent[destProcess]++;
        }
        // *** TRANSPOSE ***
        // maintain size of blocks
        int blockReceive[10] = {0};
        // send i-th blocks from each processor exactly to i-th process
        for (int i = 0; i < world_size; i++) {
          MPI_Isend( & blockSent[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, & request);

          int buf; //received size of bucket
          /* as we sent exactly to i-th process and it is not important from which source it will come, I used MPI_ANY_SOURCE, in that case status.MPI_SOURCE clould be used to determine the sender
          */
          MPI_Recv( & buf, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, & status);
          blockReceive[status.MPI_SOURCE] = buf;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // blocks redistribution
        for (int i = 0; i < world_size; i++) {
          MPI_Isend(sendBlocks[i], blockSent[i] * 2, MPI_INT, i, 0, MPI_COMM_WORLD, & request);
          MPI_Recv(receiveBlocks[i], blockReceive[i] * 2, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, & status);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // build redistributed arrray from sent blocks
        for (int sender = 0; sender < world_size; sender++) {
          for (int i = 0; i < blockReceive[sender]; i++) {
            val = receiveBlocks[sender][2 * i];
            localDestIndex = receiveBlocks[sender][2 * i + 1];
            redistributed[localDestIndex] = val;
          }
        }

        // set the values in local_input to the values in redistribudeted
        for (int i = 0; i < local_size; i++) {
          local_input[i] = redistributed[i];
        }
          free(redistributed);
      }

      int * output;
      if (world_rank == 0) {
        //for placing input array with zero padding
        output = (int * ) malloc(nadj * sizeof(int));
      }
        
      // gather all local_inputs to collect data in output array
      MPI_Gather(local_input, local_size, MPI_INTEGER, & output[world_rank * local_size], local_size, MPI_INTEGER, 0, MPI_COMM_WORLD);

      if (world_rank == 0) {
        // all artificially added zeros are ignored by incrementing the output pointer by the number of such zeros
        output += tailOffset;


        // all work by the algorithm is done, so end the time
        double time_work = MPI_Wtime();
        elapsed_time = time_work - start;
        total += elapsed_time;

        // check correctness of algorithm
        check(input, output, n);

        // before freeing output, decrement its pointer to its original location
        output -= tailOffset;
        free(output);
        free(adjustedInput);

        // if this is the final run for a given array size, average the runs and print it out
        if (run == runs - 1)
        {
          printf("Average the runs %d elements, %f sec \n", size, total / runs);
        }
        free(input);
      }
      MPI_Barrier(MPI_COMM_WORLD);

      for (int i = 0; i < world_size; i++) {
        free(sendBlocks[i]);
        free(receiveBlocks[i]);
      }

      free(sendBlocks);
      free(receiveBlocks);
      free(local_input);
      //free(redistributed);
    }
  }
  MPI_Finalize();
}
