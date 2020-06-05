///usr/local/Cellar/open-mpi/4.0.3/bin/mpic++ -o file file.c
///usr/local/Cellar/open-mpi/4.0.3/bin/mpirun --oversubscribe -np 4 file


#include <math.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <algorithm>
#include <functional>

//function that distributes arrays among processes that considers the most equal distribution possible
//output is stored in the return_array:
//return_array[0] - number of rows that the process works on
//return_array[1] - index of starting row for the process
void load_balance(int * ptr, int num_elements, int rank, int comm_size, int * return_array)
{	//if number of rows in a matrix is divisible by the number of processes then just divide the rows equally
	if (num_elements % comm_size == 0)
	{
		return_array[0] = num_elements/comm_size;
		return_array[1] = rank * return_array[0];
	}
	else
	{
		//if number of processes is bigger than number of rows, only the first num_elements processes receive one row each to work on, having one row divided among several processes results in too much overhead even for relatively big matrices
		if (rank > num_elements)
		{
			return_array[0] = 0;
			return_array[1] = num_elements;
		}
		//if number of rows in a matrix is not divisible by the number of processes then residue is divided equally among the first residue number of processes
		else if (rank < (num_elements % comm_size))
		{
			return_array[0] = num_elements/comm_size + 1;
			return_array[1] = rank * return_array[0];
		}
		else
		{
			return_array[0] = num_elements/comm_size;
			return_array[1] = (num_elements%comm_size)*(num_elements/comm_size + 1) + (rank - (num_elements % comm_size)) * return_array[0];
		}
	}
}
int randomInRange(int min, int max) {
  return rand() % (max - min + 1) + min;
}
//first command line argument specifies name of the file that contains the input matrix
//second command line argument specifies name of the output file
//if command line arguments are not specified then the program generates 10x10 random matrix and outputs the resulting matrix to the terminal
//you can arbitrarily change the size of the random matrix by changing the value of total_size variable
int main(int argc, char **argv)
{
	
	int rank;
	int comm_size;
	MPI_Init(&argc, &argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	double startTime;
	if (rank == 0) startTime = MPI_Wtime();
	
	MPI_Request request;
	int total_size;
	int * array;
	
	if (argc > 1)
	{
		FILE* f = fopen(argv[1],"r");
		//first number in the input file should specify the total number of elements in the input matrix
		//the rest of the numbers in the input file are the elements of the matrix, which is filled in row major order
		fscanf (f,"%d",&total_size);
		//master node reads the input file
		if (rank == 0)
		{
			array = (int*)malloc(total_size*sizeof(int));
			for (int i = 0; i < total_size; i++) fscanf(f,"%d", &array[i]);
		}
		fclose(f);
	}
	else
	{
		total_size = 1048576;
		if (rank == 0)
		{
			srand(time(NULL));
			array = (int*)malloc(total_size*sizeof(int));
            for (int i = 0; i < total_size; i++) {
                //array[i] = rand() % 100;
                array[i] = randomInRange(10000000, 99999999);
                
                
            }
                /*for (int i = 0; i < 10; i++)
                {
                    for (int j = 0; j < 10; j++) printf("%d ", array[i*10+j]);
                    printf("\n");
                }
                 */
            
		}
	}
	//number of rows or columns in a square matrix
	int length_of_side = sqrt(total_size);
	
	//local_range[0] stores number of rows allocated to a given process
	//local_range[1] stores the starting index of the starting row for the given process
	int local_range[2];
	load_balance(array, length_of_side, rank, comm_size, local_range);
 
 	//array that is used to make sure that correct process receives the transposed rows
	int local_receiver[local_range[0]];
	for (int i = 0; i < local_range[0]; i++) local_receiver[i] = rank;
	
	//ids of process that are supposed to receive the given transposed row during data exchange
	int receiver[length_of_side];
	//global storage for index of the starting row of each process
	int process_start [comm_size];
	//global storage for number of rows allocated to each process
	int process_length [comm_size];
	
	//let every process know the index of the starting row of each process
	MPI_Barrier(MPI_COMM_WORLD);
	double mpi_time_start;
	//compute total time of MPI calls
	if (rank == 0) mpi_time_start = MPI_Wtime();
	MPI_Allgather(&local_range[1], 1, MPI_INT, process_start, 1, MPI_INT, MPI_COMM_WORLD);
	//let every process know the number of rows allocated to each process
	MPI_Allgather(&local_range[0], 1, MPI_INT, process_length, 1, MPI_INT, MPI_COMM_WORLD);
	//the array of strides for receiving data from every process
	int strides[comm_size];
	strides[0] = 0;
	for (int i = 1; i < comm_size; i++) strides[i] = strides[i-1] + process_length[i-1];
	
	//let every pocess know the ids of process that are supposed to receive the given transposed row during data exchange
	MPI_Allgatherv(local_receiver, local_range[0], MPI_INT, receiver, process_length, strides, MPI_INT, MPI_COMM_WORLD);
	
	//local array where each process stores data allocated to it
	int * local_array =  (int*)malloc(length_of_side * local_range[0] * sizeof(int));
	
	//***************************************************INITIAL*DATA*DISTRIBUTION*BEGIN******************************************************************
	if (rank == 0)
	{
		
		for ( int i = 1; i < comm_size; i++) MPI_Isend(&array[process_start[i]*length_of_side], process_length[i]*length_of_side, MPI_INT, i, 0, MPI_COMM_WORLD, &request);  //неблокирующая отправка
		for (int i = 0; i < local_range[0]*length_of_side; i++) local_array[i] = array[i];
	}
	else
	{
		MPI_Recv(local_array, local_range[0]*length_of_side, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0)
	{
		printf("\npreprocessing time : %f seconds\n", MPI_Wtime()-startTime);
		startTime = MPI_Wtime();
	}
	//***************************************************INITIAL*DATA*DISTRIBUTION*END********************************************************************
	
	//create buffer for transposing data
	int * local_buffer = (int*)malloc(length_of_side * local_range[0] * sizeof(int));
	
	//**************************************************THE*MAIN*LOOP*ITERATION*START*******************************************************************
	//int total_time = std::log2(total_size)+1;
    int total_time = log2(total_size)+1;
	if (total_time % 2 == 1) total_time ++;
	for (int t = 0; t < total_time; t++)
	{
		for (int i = 0 ; i < local_range[0]; i++)
		{
			//strating from 0 even rows are sorted in the ascending order, odd rows are soted in the descending order
			//if we sort columns (after transposition, which is indicated by t%2==1 returning true) then sort everything in the ascending order
			if ( ((local_range[1]+i) % 2 == 1) && t%2==0)
                //sort(firstelement , last element, comparison func)
                std::sort(&local_array[i*length_of_side], &local_array[(i+1)*length_of_side], std::greater<int>());
			else std::sort(&local_array[i*length_of_side], &local_array[(i+1)*length_of_side]); //ascending order
		}
		//transpose data after each sorting - for sorting columns
		for (int i = 0; i < length_of_side; i++)
		{
			for (int j = 0; j < local_range[0]; j++) local_buffer[i*local_range[0]+j] = local_array[j*length_of_side+i];
		}
		int j = 0;
		//data exchange of transposed data - as for now data is stored in local_buffer
		for (int i = 0; i < length_of_side; i++)
		{
			MPI_Gatherv(&local_buffer[i*local_range[0]], local_range[0], MPI_INT, &local_array[j*length_of_side], process_length, strides, MPI_INT, receiver[i], MPI_COMM_WORLD);
			if (rank == receiver[i]) j++; // [0,0,1,1, ...]
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	//***************************************************THE*MAIN*LOOP*ITERATION*END**********************************************************************
	if (rank == 0)
	{
		printf("\ncomputation time : %f seconds\n", MPI_Wtime()-startTime);
		startTime = MPI_Wtime();
	}
	//process with rank 0 collects data from each process and outputs the sorted array
	int total_process_length [comm_size];
	int total_strides [comm_size];
	
	total_process_length[0] = process_length[0]*length_of_side;
	total_strides[0] = 0;
	for (int i = 1; i < comm_size; i++)
	{
		total_process_length[i] = process_length[i]*length_of_side;
		total_strides[i] = total_strides[i-1] + total_process_length[i-1];
	}
	MPI_Gatherv(local_array, local_range[0]*length_of_side, MPI_INT, array, total_process_length, total_strides, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) printf("total time of MPI calls is: %f seconds\n", MPI_Wtime() - mpi_time_start);
	if (rank == 0)
	{
		//if output file is provided then print the result to the specified file
		if  (argc > 2)
		{
			FILE * f = fopen(argv[2], "w");
			fprintf(f, "the result of computations is the following:\n");
			for (int i = 0; i < length_of_side; i++)
			{
				for (int j = 0; j < length_of_side; j++) fprintf(f, "%d ", array[i*length_of_side+j]);
				fprintf(f, "\n");
			}
			fclose(f);
		}
		//if output file is not provided then print the result to the terminal as long as the size of the array is less than 100, otherwise it's hard to read output, for that case use files
		else if (total_size <= 100)
		{	
			printf("\nthe result of computations is the following:\n");
			for (int i = 0; i < length_of_side; i++)
			{
				for (int j = 0; j < length_of_side; j++) printf("%d ", array[i*length_of_side+j]);
				printf("\n");
			}
		}
	}
	
	//free all of the dynamicallly allocated memory
	free(local_buffer);
	free(local_array);
	if(rank == 0) free(array);
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0)
	{
		printf("\npostprocessing time : %f seconds\n", MPI_Wtime()-startTime);
		startTime = MPI_Wtime();
	}
	MPI_Finalize();
	return 0;
}


