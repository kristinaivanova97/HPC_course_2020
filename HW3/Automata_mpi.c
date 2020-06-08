//
//  /usr/local/Cellar/open-mpi/4.0.3/bin/mpicc -o mat Automata2.c
//  /usr/local/Cellar/open-mpi/4.0.3/bin/mpirun --oversubscribe -np 3 mat <rule in Wolfram format>
//

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int rules (int a, int b, int c, int * ruleset){
    if      (a == 1 && b == 1 && c == 1) return ruleset[0];
    else if (a == 1 && b == 1 && c == 0) return ruleset[1];
    else if (a == 1 && b == 0 && c == 1) return ruleset[2];
    else if (a == 1 && b == 0 && c == 0) return ruleset[3];
    else if (a == 0 && b == 1 && c == 1) return ruleset[4];
    else if (a == 0 && b == 1 && c == 0) return ruleset[5];
    else if (a == 0 && b == 0 && c == 1) return ruleset[6];
    else if (a == 0 && b == 0 && c == 0) return ruleset[7];
    return 0;
}

//generate array with 0 and 1
void InitArray(int size, int * array)
{
    //random array
    srand(time(NULL));
    for(int i = 0; i < size; i++) array[i] = rand() % 2;
    
    // constant
    //for(int i = 0; i < size; i++) array[i] = 0;
    //array[size/2] = 1;
}

void generate(int size, int * prev_array, int * ruleset, int * new_array, int gost_right, int gost_left) {

    //constant boundary condition
    //new_array[0] = rules(1, prev_array[0], prev_array[1], ruleset);
    //new_array[size - 1] = rules(prev_array[size - 2], prev_array[size-1], 1, ruleset);
        
    new_array[0] = rules(gost_left, prev_array[0], prev_array[1], ruleset);
    new_array[size - 1] = rules(prev_array[size - 2], prev_array[size-1], gost_right, ruleset);
    
    for (int i = 1; i < size - 1; i++) {
      int left   = prev_array[i-1];
      int me     = prev_array[i];
      int right  = prev_array[i+1];
      new_array[i] = rules(left, me, right, ruleset);
    }

  }

static void set_rule_binary(int rule, int * rule_binary)
{
    for(int p = 0; p <= 7; p++)
    {
        if((int)(pow(2, p)) & rule)
        {
            rule_binary[abs(p - 7)] = 1;
        }
        else
        {
            rule_binary[abs(p - 7)] = 0;
        }
    }
}

void load_balance(int size, int rank, int comm_size, int* local_size, int* place)
{
    if (size % comm_size == 0)
    {
        local_size[0] = size/comm_size;
        place[0] = rank*local_size[0];

    }
    else
    {
        if (rank > size)
        {
            local_size[0] = 0;
            place[0] = size;

        }
        else if (rank < (size % comm_size))
        {
            local_size[0] = size/comm_size + 1;
            place[0] = rank*local_size[0];
        }
        else
        {
            local_size[0] = size/comm_size;
            place[0] = (size%comm_size)*(size/comm_size + 1) + (rank - (size % comm_size)) * local_size[0];
        }
    }
}

int main(int argc, char ** argv)
{
    
    int rank;
    int comm_size;
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Request request, request2;
    
    int size = 10; //size of an array
       
    int rounds = 10;
    int rule; // = 54;
    int * rule_binary =  (int*)malloc(sizeof(int)*8);
    
    rule = atoi(argv[1]); //Enter rule in Wolfram format
    set_rule_binary(rule, rule_binary);

    int * array;
    int * for_print;
    
    if (rank == 0) {
        array = (int*)malloc(size*sizeof(int));
        InitArray(size, array);
        for_print = (int*)malloc(size*sizeof(int));
        printf("Initial array: \n");
        for (int i = 0; i < size; i++)
        {
           printf("%d", array[i]);
        }
        printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
        
    //*** Communication part ***
    
    //MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //MPI_Bcast(&rounds, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //MPI_Bcast(rule_binary, 8, MPI_INT, 0, MPI_COMM_WORLD);

    int local_size[1];
    int place[1];
    load_balance(size, rank, comm_size, local_size, place);
    
    int * local_array =  (int*)malloc(local_size[0] * sizeof(int));
    int process_length [comm_size];
    int process_place [comm_size];
    //let every process know the number of elems allocated to each process
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allgather(&local_size, 1, MPI_INT, process_length, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&place, 1, MPI_INT, process_place, 1, MPI_INT, MPI_COMM_WORLD);
    
    
    
    int strides[comm_size];
    strides[0] = 0;
    for (int i = 1; i < comm_size; i++) strides[i] = strides[i-1] + process_length[i-1];
    
    MPI_Scatterv(array, process_length, strides, MPI_INT, local_array, local_size[0],
                 MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    double mpi_time_start;
    if (rank == 0) mpi_time_start = MPI_Wtime();
    
    int * new_array = (int*)malloc(local_size[0]*sizeof(int));
    int gost_left[1], gost_right[1];
    
    // *** Sending and recieving gosts ***
    for(int m = 0; m < rounds; m++)
    {

        MPI_Isend(&local_array[0], 1, MPI_INT, (rank-1+ comm_size)%comm_size, 0,  MPI_COMM_WORLD, &request);
        MPI_Irecv(gost_right, 1, MPI_INT, (rank+1)%comm_size, 0, MPI_COMM_WORLD, &request);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        MPI_Isend(&local_array[local_size[0]-1], 1, MPI_INT, (rank+1)%comm_size, 1,  MPI_COMM_WORLD, &request2);
        MPI_Irecv(gost_left, 1, MPI_INT, (rank-1+ comm_size)%comm_size, 1, MPI_COMM_WORLD, &request2);
        
        MPI_Barrier(MPI_COMM_WORLD);
        //printf("My left = %d, my right = %d \n", gost_left[0], gost_right[0]);
        
        generate(local_size[0],local_array, rule_binary, new_array, gost_right[0], gost_left[0]);
        //memcpy(&array, &new_array, size);
        MPI_Barrier(MPI_COMM_WORLD);
        for (int h = 0; h < size; h++)
        {
            local_array[h] = new_array[h];
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gatherv(local_array, local_size[0], MPI_INT, for_print, process_length, strides, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
    
            for(int j = 0; j< size; j++)
            {
                if (for_print[j] == 0) {printf(" ");}
                else if (for_print[j] == 1) {printf("*");}
                else {printf("error ");}
            }
            printf("|");
            printf("\n");
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("total time of MPI calls is: %f seconds\n", MPI_Wtime() - mpi_time_start);
     
    if (rank == 0)
    {
        free(array);
        free(for_print);
    }
    free(local_array);
    free(rule_binary);
    free(new_array);
    MPI_Finalize();
}

// IN serial version i'v got

/* for periodic boundary conditions
 rule: 0 1 0 1 1 0 1 0
 initial array:
 000000000000000100000000000000
              * *             |
             *   *            |
            * * * *           |
           *       *          |
          * *     * *         |
         *   *   *   *        |
        * * * * * * * *       |
       *               *      |
      * *             * *     |
     *   *           *   *    |
    * * * *         * * * *   |
   *       *       *       *  |
  * *     * *     * *     * * |
 *   *   *   *   *   *   *   *|
  * * * * * * * * * * * * * * |
 
same for static cond = 1:
 Initial array:
 000000000000000100000000000000
 *             * *            *|
 **           *   *          **|
  **         * * * *        ** |
  ***       *       *      *** |
  * **     * *     * *    ** * |
    ***   *   *   *   *  ***   |
 * ** ** * * * * * * * *** ** *|
 * ** **               * * ** *|
 * ** ***             *    ** *|
 * ** * **           * *  *** *|
 * **   ***         *   *** * *|
 * *** ** **       * * ** *   *|
 * * * ** ***     *    **  * **|
 *     ** * **   * *  *****  * |
 **   ***   *** *   ***   ***  |
  ** ** ** ** *  * ** ** ** ***|
  ** ** ** **  **  ** ** ** *  |
  ** ** ** ********** ** **  **|
 
 Initial array:
 Periodic boundaries.
 Rule : 1 0 1 1 0 1 1 0
 00000000000000000000000001000000000000000000000000
                         ***                       |
                        * * *                      |
                       *******                     |
                      * ***** *                    |
                     *** *** ***                   |
                    * * * * * * *                  |
                   ***************                 |
                  * ************* *                |
                 *** *********** ***               |
                * * * ********* * * *              |
               ******* ******* *******             |
              * ***** * ***** * ***** *            |
             *** *** *** *** *** *** ***           |
            * * * * * * * * * * * * * * *          |
           *******************************         |
          * ***************************** *        |
 
*/

/*
 Initial array randomly initialized:
 
1110011011100111100101001010000000111100
 * **  * * ** ** ***********     * ** **|
***  ******  *  * ********* *   ***  *  |
 * ** **** ******* ******* *** * * *****|
***  * ** * ***** * ***** * * ***** *** |
 * ****  *** *** *** *** ***** *** * * *|
*** ** ** * * * * * * * * *** * * ******|
** *  *  ***************** * ***** *****|
* ******* *************** *** *** * ****|
 * ***** * ************* * * * * *** ***|
*** *** *** *********** ********* * * * |
 * * * * * * ********* * ******* *******|
************* ******* *** ***** * ***** |
 *********** * ***** * * * *** *** *** *|
* ********* *** *** ******* * * * * * **|
 * ******* * * * * * ***** *********** *|
*** ***** *********** *** * ********* **|
** * *** * ********* * * *** ******* * *|
* *** * *** ******* ***** * * ***** *** |
** * *** * * ***** * *** ***** *** * * *|
* *** * ***** *** *** * * *** * * ***** |
 
*/
